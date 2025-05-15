import os
import torch
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights
model_path = "/share-data/wzk-1/model/deepseek-v2-lite"
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

# 获取模型层结构
layer_names = [name for name, _ in model.named_modules()]
print("Model layers:", layer_names)