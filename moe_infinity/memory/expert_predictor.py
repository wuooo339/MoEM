from transformers import PretrainedConfig
import math
from moe_infinity.memory.expert_tracer import ExpertTracer
from moe_infinity.utils import parse_moe_param


class ExpertPredictor:
    def __init__(self, config: PretrainedConfig) -> None:
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.layer_decay_func = lambda x, l: -1 / math.exp(x - l + 1)
    def add_tracer(self, tracer: ExpertTracer):
        self.tracer = tracer
    def predict(self, seq_id, expert_list, layer_idx,k = 3):
        # 进行计数更新，使用LFU策略，每次路由进行预测一次
        # 预测从当前层的可能性
        self.tracer.update_entry(seq_id, expert_list, layer_idx)
        current_entry = self.tracer.get_entry(seq_id)
        # start_time = time.time()
        expert_matrix = self.tracer.get_last(current_entry.matrix, layer_idx,0)
        # expert_matrix = self.tracer.find_most_similar(current_entry.matrix, layer_idx)
        # print("find_most_similar", time.time() - start_time)
        # expert_matrix = copy.deepcopy(entry)
        # expert_matrix[ :layer_idx, :] = 0
        # for l in range(layer_idx, min(layer_idx + k, self.num_layers)):
        #     # expert_matrix[l] = (expert_matrix[l] + 1e-8) * self.layer_decay_func(layer_idx, l)
        #     print(expert_matrix[l, :])
        return expert_matrix