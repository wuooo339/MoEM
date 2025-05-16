# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from moe_infinity.utils import ArcherConfig
from moe_infinity.memory.global_prefetch import is_expert_prefetched ,format_prefetched_experts,get_prefetched_output
DEBUG = False
def _call_expert_dispatcher(method, *args, **kwargs):
    global _expert_dispatcher
    func = getattr(_expert_dispatcher, method)
    return func(*args, **kwargs)


class DistributedExpertExecutor:
    def __init__(self, archer_config: ArcherConfig):
        self.archer_config = archer_config

    def set_expert_dispatcher(self, expert_dispatcher):
        global _expert_dispatcher
        _expert_dispatcher = expert_dispatcher
        self.expert_dispatcher = expert_dispatcher

    def set_device_map_manager(self, device_map_manager):
        self.device_map_manager = device_map_manager

    def dispatch_local(self, hidden_states, router_mask, layer_id):
        num_expert = router_mask.shape[-1]
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )
        total_gpus = torch.cuda.device_count()
        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )
        expected_wait_cnt = len(expert_list)
        hits = []
        hit = 0
        # print(f"\033[95m ===Dispatch Local in Layer{layer_id}===:【{expected_wait_cnt}】-{expert_list}\033[0m")
        self.expert_dispatcher.set_inputs(hidden_states, router_mask)    
        if layer_id % 6 == 0:
            # 到达预取窗口层
            prefetch_info = format_prefetched_experts(layer_id + 3)
            print(prefetch_info)
        if layer_id % 6 == 3:
            # 到达预取执行层
            for expert_id in expert_list:
                gpu_id = expert_id % total_gpus
                if is_expert_prefetched(layer_id,expert_id):
                    hits.append(expert_id)
                    hit=1
                    print(f"\033[1;33m╔══════════════════════════╗\033[0m")
                    print(f"\033[1;33m║ 🚀 预取命中: L{layer_id}-E{expert_id}    ║\033[0m")
                    print(f"\033[1;33m╚══════════════════════════╝\033[0m")
        self.expert_dispatcher.set_expected_queue(expected_wait_cnt)
        
        for expert_id in expert_list:
            gpu_id = expert_id % total_gpus
            self.expert_dispatcher.enqueue_expert(layer_id, expert_id, gpu_id, False)
        result = self.expert_dispatcher.wait_expert()
        return result


    def dispatch(self, hidden_states, router_mask, layer_id):
        num_expert = router_mask.shape[-1]
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )

        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )
        print("\033[95m===dispatch===\033[0m")

        device_list = self.device_map_manager.get_target_device(expert_list)
        visited_ranks = set()
        rank_wait_cnt = {r: 0 for r in range(dist.get_world_size())}
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            visited_ranks.add(rank)
            rank_wait_cnt[rank] += 1

        futures = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_inputs", hidden_states.cpu(), router_mask.cpu()),
                )
                futures.append(future)
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_expected_queue", rank_wait_cnt[rank]),
                )
                futures.append(future)
            else:
                self.expert_dispatcher.set_inputs(hidden_states, router_mask)
                self.expert_dispatcher.set_expected_queue(rank_wait_cnt[rank])

        # wait for all futures
        for future in futures:
            future.wait()

        futures = []
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            if rank == dist.get_rank():
                self.expert_dispatcher.enqueue_expert(
                    layer_id, expert_id, gpu_id, False
                )
            else:
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("enqueue_expert", layer_id, expert_id, gpu_id, True),
                )
                futures.append(future)

        # wait for all futures
        for future in futures:
            future.wait()

        result_list = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                result = rpc.rpc_sync(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("wait_expert",),
                )
                result_list += result
            else:
                result = self.expert_dispatcher.wait_expert()
                result_list += result

        return result_list
