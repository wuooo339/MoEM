import copy
import os
import uuid
from collections import Counter
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# from sklearn.metrics.pairwise import cosine_similarity
from moe_infinity.memory.expert_entry import ExpertTraceEntry
from moe_infinity.utils import parse_moe_param


class ExpertTracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExpertTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self, capacity: int, config: PretrainedConfig):
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        # self.capacity = capacity
        self.capacity = 10
        self.next_seq_id = 0 
        self.trace = {}

        # self.trace_collection = torch.zeros((capacity, self.num_layers, self.num_experts), device="cuda:0")
        self.trace_collection = {}
        self.collection_access = np.zeros((capacity,))
        self.seq_count = 0
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def load_trace(self, trace: Union[os.PathLike, np.ndarray]):
        if isinstance(trace, os.PathLike):
            self.trace_collection = torch.from_numpy(
                np.load(trace, allow_pickle=False)
            )
        elif isinstance(trace, np.ndarray):
            self.trace_collection = trace

        self.persistent_capacity = self.trace_collection.shape[0]
        assert self.persistent_capacity <= self.capacity, (
            f"loaded trace capacity {self.persistent_capacity} must be "
            f"less than or equal to capacity in config {self.capacity}"
        )

    def create_entry(self):
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.trace[seq_id] = ExpertTraceEntry(seq_id, np.zeros((self.num_layers, self.num_experts)), 0, 0)
        self.trace_collection[seq_id] = ExpertTraceEntry(seq_id, np.zeros((self.num_layers, self.num_experts)), 0, 0)
        return seq_id

    def finish_entry(self, seq_id):
        trace_sum = np.sum(self.trace_collection, axis=(1, 2))

        if np.any(trace_sum == 0):
            # find the first zero entry
            idx = np.argwhere(trace_sum == 0)[0][0]
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = 1
        else:
            # find the first entry after self.persistent_capacity that has the least access
            collection_access_copy = self.collection_access.copy()
            collection_access_copy[: self.persistent_capacity] = 1e9

            idx = np.argmin(collection_access_copy)
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = 1

    def update_entry(self, seq_id, expert_list, layer_idx):
        expert_counter = Counter(expert_list.flatten().tolist())
        # 更新专家计数
        for key, count in expert_counter.items():
            self.trace[seq_id].matrix[layer_idx, key] += count
            print(f"[{key}]:{self.trace[seq_id].matrix[layer_idx, key]}",end=" | ")
        print(f"\n---------------------------------------")
        # 如果是最后一层且达到保存条件
        if layer_idx == self.num_layers - 1:
            self.trace[seq_id].num_new_tokens += 1
            if self.trace[seq_id].num_new_tokens % 32 == 0:  # 保存矩阵记录
                idx = self.seq_count % self.capacity
                matrix = self.trace[seq_id].matrix
                print("原始矩阵 (layer × expert):")
                print(f"形状: {matrix.shape} (层数: {matrix.shape[0]}, 专家数: {matrix.shape[1]})")
                # 逐层打印非零专家（避免打印全零层）
                for l in range(matrix.shape[0]):
                    non_zero_experts = np.where(matrix[l] != 0)[0]
                    if len(non_zero_experts) > 0:
                        print(f"  层 {l:2d} - 活跃专家: {non_zero_experts}")
                self.trace_collection[idx].matrix = matrix
                self.collection_access[idx] = 1
                self.seq_count += 1
    
    def get_entry_decoder(self, seq_id):
        entry = copy.deepcopy(self.trace[seq_id])
        entry.matrix[: self.num_encoder_layers, :] = 0
        return entry

    def get_entry(self, seq_id):
        return self.trace[seq_id]

    def find_most_similar(self, matrix: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        根据余弦相似度找到最相似的 ExpertTraceEntry
        Args:
            matrix: 当前请求的专家权重矩阵 (num_layers, num_experts)
            layer_idx: 当前处理的层索引
        Returns:
            最相似的历史条目矩阵 (num_layers, num_experts)
        """
        if self.seq_count <= 0 or not self.trace_collection:
            return matrix

        try:
            # 1. 准备输入数据
            current_matrix = torch.from_numpy(matrix).float().to("cuda:0")
            
            # 2. 收集所有历史矩阵并预处理
            history_matrices = []
            valid_indices = []   
            for idx, entry in enumerate(self.trace_collection.values()):
                if not isinstance(entry.matrix, np.ndarray):
                    continue
                # 复制并处理历史矩阵
                hist_matrix = torch.from_numpy(entry.matrix.copy()).float().to("cuda:0")
                hist_matrix[layer_idx, :] = 1e-9  # 防止当前层干扰
                history_matrices.append(hist_matrix)
                valid_indices.append(idx)
            if not history_matrices:
                return matrix
            # 3. 堆叠历史矩阵 (num_entries, num_layers, num_experts)
            history_stack = torch.stack(history_matrices)
            # 4. 归一化处理
            def safe_normalize(x):
                x_sum = torch.sum(x, dim=-1, keepdim=True)
                x_sum = torch.where(x_sum == 0, torch.ones_like(x_sum), x_sum)
                return x / x_sum
            current_norm = safe_normalize(current_matrix)
            history_norm = safe_normalize(history_stack)
            # 5. 计算余弦相似度
            current_expanded = current_norm.unsqueeze(0)  # (1, num_layers, num_experts)
            sim_matrix = self.cos(current_expanded, history_norm)  # (num_entries, num_layers)

            # 6. 计算平均相似度并选择最佳匹配
            avg_sim = torch.mean(sim_matrix, dim=1)  # (num_entries,)
            best_idx = torch.argmax(avg_sim).item()
            best_entry_idx = valid_indices[best_idx]
            # 7. 更新访问计数
            best_entry = list(self.trace_collection.values())[best_entry_idx]
            print(f"\n[DEBUG] 最佳匹配条目{best_entry_idx}:")
            print("逐层专家权重:")
            for layer in range(best_entry.matrix.shape[0]):
                experts_str = ", ".join([f"{w:.6f}" for w in best_entry.matrix[layer]])
                layer_mark = "->" if layer == layer_idx else "  "
                print(f"{layer_mark} Layer {layer}: [{experts_str}]")
            return best_entry.matrix
        except Exception as e:
            print(f"[ERROR] 相似度计算失败: {str(e)}")
            return matrix
    def get_last(self, matrix, layer_idx, n=0) -> np.ndarray:
        idx = max(self.seq_count - 1,0)
        ratio = 0.8
        if(self.seq_count > n):
            print(f"return EAMC[{idx}]")
            #逐元素乘法：保留的历史信息占比
            return self.trace_collection[idx].matrix*(1-ratio)+matrix*ratio
        else:
            print("return iEAM")
            entry = matrix
        return entry