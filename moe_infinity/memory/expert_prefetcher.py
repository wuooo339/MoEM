# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


import numpy as np
from transformers import PretrainedConfig

from moe_infinity.utils import parse_moe_param
from moe_infinity.memory.global_prefetch import add_prefetched_expert

class ExpertPrefetcher(object):
    cache_file_rd = None
    first_k_dense_replace: int = 0

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )

    def set_archer_engine(self, archer_engine):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def prefetch_experts_list(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def fetch_experts_lock_cache(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts(self, layer_id, expert_matrix):
        expert_list = []
        # print("expert_tensor_map", self.expert_tensor_map)
        for i in range(layer_id, self.num_layers):
            for j in range(self.num_experts):
                if expert_matrix[i, j] > 0:
                    expert_list.append((self.expert_tensor_map[(i, j)], expert_matrix[i, j]))
        ordered_expert_list = sorted(expert_list, key=lambda x: x[1], reverse=True)
        tensor_ids = [x[0] for x in ordered_expert_list]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
     # 测试引擎，方便调试
    def prefetch_experts_test(self, fetch_layer_id, expert_matrix, k=1):
        """预取专家模块数据（美化输出版）"""
        # ==================== 打印预取头部信息 ====================
        header = "=" * 20 + " Prefetch Experts " + "=" * 20
        print(f"\n\033[1;36m{header}\033[0m")  # 青色加粗标题
        
        expert_list = []
        for i in range(fetch_layer_id, min(fetch_layer_id + k, self.num_layers)):
            # ==================== 当前层信息 ====================
            print(f"\n\033[1;33mFetch Layer [{i}]:\033[0m")
            current_layer_experts = []
            current_layer_fetch = []
            for j in range(self.num_experts):
                current_layer_experts.append((i, j, expert_matrix[i, j]))
                current_layer_fetch.append((self.expert_tensor_map[(i, j)], expert_matrix[i, j]))
            # 按权重排序取Top32
            ordered_experts = sorted(current_layer_experts, key=lambda x: x[2], reverse=True)[:32]
            ordered_fetch = sorted(current_layer_fetch, key=lambda x: x[1], reverse=True)[:32]
            # ==================== 打印专家权重 ====================
            print("\033[90mTop 32 Experts:\033[0m")  # 灰色小字注释
            for idx, (_, expert, weight) in enumerate(ordered_experts, 1):
                color = "\033[32m" if weight > 0.5 else "\033[33m"  # 高权重绿色，低权重黄色
                print(f"{color}Expert {expert:2d}: {weight:.3f}\033[0m", end=" | ")
                if idx % 4 == 0:  # 每行显示4个专家
                    print()
            print("\n" + "-" * 60)
            expert_list.append(ordered_fetch)
        
        # ==================== 实际预取操作 ====================
        print("\n\033[1;35m" + "="*20 + " Real Fetch Operation " + "="*20 + "\033[0m")
        keys = [((fetch_layer_id) % 26, 19), ((fetch_layer_id ) % 26, 31)]
        for layer_id, expert_id in keys:
            add_prefetched_expert(layer_id, expert_id)
        tensor_ids = [self.expert_tensor_map[key] for key in keys]
        # 打印带颜色的tensor信息
        print("\033[94mUnique Tensor IDs:\033[0m", np.unique(tensor_ids))
        print("\033[94mAll Tensor IDs:\033[0m   ", tensor_ids)
        # 执行预取
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
            print(f"Prefetching \033[1;36mTensor {tensor_id}\033[0m → GPU {gpu_id}")
        # ==================== 结束标记 ====================
        success_banner = "\n\033[1;32m" + "✓"*10 + " Fetch Success! " + "✓"*10 + "\033[0m"
        print(success_banner)