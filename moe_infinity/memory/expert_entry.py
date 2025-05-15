from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertTraceEntry:
    def __init__(self, seq_id: int, matrix: np.ndarray, access: int, num_new_tokens: int):
        self.seq_id = seq_id  # 明确类型为 int
        self.matrix = matrix
        self.access = access
        self.num_new_tokens = num_new_tokens

    def __hash__(self):
        return hash(self.seq_id)  # 直接哈希整数


@dataclass
class ExpertCacheEntry:
    expert_idx: int = None
    layer_idx: int = None
    r: float = 0.0
    visit: int = 0
    timestamp: int = 0

    def __hash__(self):
        return hash((self.layer_idx, self.expert_idx))
