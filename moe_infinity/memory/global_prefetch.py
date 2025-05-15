# global_prefetch.py
from typing import Dict, Set, Tuple, Optional

_prefetched_experts: Dict[int, Set[int]] = {}

def get_prefetched_output(layer_id: int):
    return layer_id
def add_prefetched_expert(layer_id: int, expert_id: int):
    """添加预取的专家，自动初始化该层的集合"""
    _prefetched_experts.setdefault(layer_id, set()).add(expert_id)

def is_expert_prefetched(layer_id: int, expert_id: int) -> bool:
    """显式检查层和专家是否存在"""
    if layer_id not in _prefetched_experts:
        return False
    return expert_id in _prefetched_experts[layer_id]

def clear_prefetched_experts(layer_id: int = None):
    """清空预取记录，可指定层或全部清空（安全版本）"""
    if layer_id is None:
        _prefetched_experts.clear()
    else:
        # 安全删除，即使key不存在也不会报错
        _prefetched_experts.pop(layer_id, None)

def get_prefetched_experts(layer_id: Optional[int] = None) -> Set[Tuple[int, int]]:
    """获取预取的专家集合"""
    if layer_id is None:
        return {(lid, eid) for lid, experts in _prefetched_experts.items() for eid in experts}
    return {(layer_id, eid) for eid in _prefetched_experts.get(layer_id, set())}

def format_prefetched_experts(layer_id: Optional[int] = None) -> str:
    """独立的美观打印函数，避免递归"""
    experts = get_prefetched_experts(layer_id)  # 这里调用的是获取集合的函数
    
    if not experts:
        return "┌─────────────┐\n│ 无预取专家 │\n└─────────────┘"
    
    # 按层分组
    layer_groups: Dict[int, Set[int]] = {}
    for lid, eid in experts:
        layer_groups.setdefault(lid, set()).add(eid)
    
    # 构建美观输出
    lines = []
    header = "┌──────────────────────────────┐"
    title = "│ 预取专家信息 (Layer/Expert) │"
    footer = "└──────────────────────────────┘"
    
    lines.append(header)
    lines.append(title)
    
    for lid in sorted(layer_groups):
        experts_str = ", ".join(f"{eid:2d}" for eid in sorted(layer_groups[lid]))
        lines.append(f"│ Layer {lid:2d}: {experts_str} │")
    
    lines.append(footer)
    return "\n".join(lines)
