import torch
import xxhash
import math
from typing import Union
def minmax_normalize(
        x: Union[float, int],
        min_val: Union[float, int],
        max_val: Union[float, int],
        feature_range: tuple = (0, 1)) -> float:
    """将x的值线性映射到指定范围内"""
    if max_val == min_val:
        raise ValueError("min_max_normalize中，max_val和min_val不能相等")
    target_min, target_max = feature_range
    x_clamped = max(min(x, max_val), min_val)
    scale = (x_clamped - min_val) / (max_val - min_val)
    normalized = scale * (target_max - target_min) + target_min
    return float(normalized)

def normal_normalize(
        x: Union[float, int],
        min_val: Union[float, int],
        p99_val: Union[float, int]) -> float:
    """通用方法，区别于知道最大值的minmax_normalize，将x映射到0-1之间
        p99_val表示99%的情况下的最大值，用于处理长尾分布的数据
        参数：
            x: 待归一化的原始值（如怪物攻击值）
            min_val: 数据的最小值（如攻击最小值0）
            p99_val: 99%的数据都会低于此值（用于替代全局最大值，处理长尾）

        返回：
            float: 归一化后的值（范围[0,1]）
    """
    if p99_val <= min_val:
        raise ValueError("normal_normalize中，p99_val必须大于min_val")
    mean = (p99_val + min_val) / 2
    std = (p99_val - min_val) / 3  # 假设99%的数据在mean±3*std范围内
    z_score = (x - mean) / std
    normalized = 1 / ( 1 + math.exp(-z_score))
    
    return float(normalized)
def get_hash_val_normalized(input_str: str) -> torch.Tensor:
    """计算字符串的hash值并映射到0 - 1之间"""
    MAX_UINT64 = 2**64 - 1
    data = input_str.encode('utf-8')
    hash_value = xxhash.xxh64(data).intdigest()
    normalized_hash = hash_value / MAX_UINT64
    return torch.tensor(normalized_hash, dtype=torch.float32).unsqueeze(0)

def _pad_vector_list(vec_list, max_n, vec_size=None, default_size=0):
    """Flatten and concatenate vectors in vec_list and pad to max_n * vec_size.

    - vec_list: iterable of torch tensors
    - max_n: target number of items
    - vec_size: per-item vector size; if None inferred from first item
    - default_size: used when vec_list empty and vec_size is None
    Returns a 1-D torch tensor of length max_n * vec_size
    """
    items = [torch.flatten(v) for v in vec_list]
    if items:
        inferred_size = items[0].numel() if vec_size is None else vec_size
        block = torch.cat(items)
        pad_len = (max_n - len(items)) * inferred_size
        if pad_len > 0:
            block = torch.cat([block, torch.zeros(pad_len)])
        return block
    else:
        size = vec_size if vec_size is not None else default_size
        return torch.zeros(max_n * size)
if __name__ == "__main__":
    # 测试normal_normalize
    print(normal_normalize(100,0,50))