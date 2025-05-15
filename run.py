#CUDA_VISIBLE_DEVICES=1 python run.py 2>&1 | tee output/log_test.txt
#CUDA_VISIBLE_DEVICES=2 ./build/bin/llama-cli -m /share-data/wzk-1/model/deepseek-v2-lite/deepseek-v2-lite.gguf -ngl 12
#CUDA_VISIBLE_DEVICES=3 ./run-llama.sh | tee output/log_llama.txt

import argparse
import multiprocessing as mp
import os
import time
import warnings
from functools import partial
import sys
warnings.filterwarnings("ignore")

import datasets
import torch
from transformers import AutoTokenizer, LlamaTokenizerFast, TextStreamer
from moe_infinity import MoE
from moe_infinity.models.modeling_arctic import ArcticTokenizer
print("CUDA_VISIBLE_DEVICES:", os.getenv('CUDA_VISIBLE_DEVICES'))
print("Available devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# 在循环外部添加计数器
processed_count = 0
max_executions = 50  # 最大执行次数
class StopWatch(TextStreamer):
    def __init__(self, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0
        self.engine = engine
        self.num_layers = engine.num_layers
        self.num_experts = engine.num_experts
    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            print("write out information.")
            self.start_decoding = time.time()
        self.decoding_iterations += 1
        # 打印Expert Cache的状态，输出会被写入文件
        # self.engine.expert_dispatcher.get_cur_state(self.num_experts, self.num_layers, "state/state.txt")
        self.engine.expert_dispatcher.set_expert_cache_priority()
        if self.decoding_iterations % 100 == 0:
            current_time = time.time()
            print(f"\n-----------------------------------------------")
            print(f"Prefilling time: {self.prefilling_time} seconds")
            print(f"Decoding time: {self.decoding_time} seconds")
            print(f"Decoding iterations: {self.decoding_iterations}")
            print(f"Decoding time per iteration: {(current_time-self.start_decoding) / self.decoding_iterations} seconds")
        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding

        return super().end()
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="/share-data/wzk-1/model/deepseek-v2-lite")
parser.add_argument("--offload_dir", type=str, default="/home/user/offload/deepseek-v2-param")
parser.add_argument("--device_memory_ratio", type=float, default=0.71)
parser.add_argument("--out_len", type=int, default=32)
args = parser.parse_args()

model_name = args.model_name_or_path.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
all_inputs = [
    "请逐步解决以下问题，并解释每步推理过程：\n1. 若3台机器5小时生产180个零件，7台机器8小时可生产多少零件？\n2. 甲比乙大6岁，5年前甲年龄是乙的2倍，求两人现在年龄。\n3. 计算：(2³ × √16) ÷ (4⁻¹ + log₂8)\n4. 一个骰子连续掷3次，至少出现一次6的概率是多少？\n5. 用贝叶斯定理解释：新冠检测准确率98%，人群感染率1%，某人检测阳性时实际患病的概率是多少？\n要求：分步骤展示计算过程，最终答案用【】标注。",
    
    "请用Python实现以下需求：\n1. 编写一个支持LRU缓存机制的装饰器类，包含get/put方法\n2. 用PyTorch构建一个3层MoE模型：\n   - 专家数=4，门控网络为简单线性层\n   - 每专家为含Dropout的全连接网络\n   - 支持动态专家激活数（top_k=2）\n3. 修复以下代码的BUG（提示：涉及异步协程）：\n   async def fetch_data():\n       results = []\n       for url in urls:\n           data = await session.get(url)  # 报错位置\n           results.append(data.json())\n       return results\n要求：代码需可直接运行，关键处添加注释。",
    
    "请完成以下跨语言任务：\n1. 将中文谚语'塞翁失马，焉知非福'翻译成英文、法文、日文，并分别给出文化背景解释\n2. 把西班牙语歌词'Despacito'（原意：慢慢来）本地化为中文四字成语风格\n3. 分析德语复合词'Schadenfreude'（幸灾乐祸）在中文/阿拉伯语中的等效表达\n4. 为日本客户撰写商务邮件（日文），主题：AI合作项目延期请求（需符合敬语规范）\n要求：译文需符合目标语言文化习惯，重要概念附加说明。",
    
    "请用学术论文风格回答：\n1. 对比Transformer/RNN/GNN在时序预测中的优劣（需引2020年后论文）\n2. 用控制论解释AlphaGo的决策树优化过程\n3. 列出量子计算对密码学的三大影响，并说明Shor算法原理\n4. 撰写摘要：MoE模型在边缘设备部署的挑战（200字内，含能耗/精度/延迟指标）\n要求：关键论点需标注参考文献（格式：Author et al., Year），避免主观表述。",
    
    "根据以下线索生成创意内容：\n1. 用'区块链+碳中和'概念设计一个DApp交互流程图\n2. 为科幻小说《月球AI殖民地》编写故事大纲（3幕剧结构）\n3. 将古诗'大漠孤烟直'转写成现代诗歌并配50字视觉画面描述\n4. 设计一个哲学悖论：当超级AI的效用函数与人类伦理冲突时（模仿电车难题）\n5. 用emoji+短句描述'元宇宙婚礼'的核心体验（限20个字符）\n要求：每个产出需包含创新性和可实施性说明。"
]
# all_inputs = [
#     "In what follows, we provide short narratives, each of which illustrates a common proverb."
#     "Narrative-1: Maria is always good to her sister Joy even though her sister doesn't even recognize it. I admire Maria's attitude for being that way even though her sister doesn't recognize it but still she believes that being good will be rewarded later on. A little bit smile on our face or helping anyone everyday is a simple thing but we don't how this touches lives to others we just don't even recognize it that we are a blessings to others even just a simple way or act of kindness."
#     "Narrative-2: Gayle worked in HR and was so mean everyone called her unit 'Inhuman Resources'.  Although it was her job to administer benefits, she took delight in not doing the necessary actions to insure employees were covered.  Soon it came to the attention of the CEO and he demanded an explanation for each and every employee she had treated wrongly.  She had to work nights and weekends for weeks on end to gather all the information for him. " 
#     "Narrative-3: There was once a poor farmer with two young adult boys.  One boy went off into the world to make his way, and left his poor father and brother alone to tend the farm by themselves.  The farm soon fell into disrepair as the work fell behind.  The boy was lost in the world, and finally went home to his father.  He asked for forgiveness for his selfish ways.  The father forgave the son for the harm that had been caused by his actions."
#     "these narratives are good illustration of the following proverbs."
# ]

config = {"offload_path": os.path.join(args.offload_dir, model_name),"device_memory_ratio": args.device_memory_ratio,}
model = MoE(args.model_name_or_path, config)

custom_kwargs = {}
if "deepseek" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
else:
    raise ValueError(f"Model {args.model_name_or_path} not supported")

tokenizer.pad_token = tokenizer.eos_token
cnt = 0
max_seq_length = 512
print(f"------------Priority Setting In Advance------------")
model.engine.expert_dispatcher.set_expert_cache_priority('priority.txt')
# with open("input.txt", "a", encoding="utf-8") as input_file:
for input_text in all_inputs:
    # repeat the input text 100 times to test the performance
    if processed_count >= max_executions:
        print(f"已达到最大执行次数 {max_executions}，停止处理")
        break
    # input_text = input_text
    print("input: ",input_text)
    # input_file.write(input_text + "\n")
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="do_not_pad",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    print("inputs ...")
    print(inputs.input_ids.shape)
    
    streamer = StopWatch(model.engine, tokenizer)
    with torch.no_grad():
        print("outputs_text ...")
        outputs = model.generate(
            inputs.input_ids.to("cuda:0"),
            streamer=streamer,
            max_new_tokens=args.out_len,
            min_new_tokens = args.out_len,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            use_cache=True,
            **custom_kwargs,
        )
        print(f"Prefilling time: {streamer.prefilling_time} seconds")
        print(f"Decoding time: {streamer.decoding_time} seconds")
        print(f"Decoding iterations: {streamer.decoding_iterations}")
        print(f"Decoding time per iteration: {streamer.decoding_time / streamer.decoding_iterations} seconds")
        print(f"Input tokens: {len(inputs.input_ids[0])}")
        processed_count += 1