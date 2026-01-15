import math
from typing import List, Dict, Tuple

def tokenize_time_series(
    values: List[float],
    num_buckets: int = 10,
    eps: float = 1e-8
) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    A test hybrid tokenizer for numeric time series.

    Args:
        values: ordered numeric time series
        num_buckets: number of quantization buckets
        eps: numerical stability constant

    Returns:
        tokens: symbolic tokens (e.g., "<NUM>")
        features: per-token numeric features
    """

    if not values:
        return [], []

    # --- basic stats ---
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) + eps

    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val + eps

    tokens = []
    features = []

    prev_value = values[0]

    for i, v in enumerate(values):
        delta = 0.0 if i == 0 else v - prev_value

        norm_value = (v - mean) / std
        log_value = math.log(abs(v) + 1.0)
        norm_delta = delta / std

        # quantization / bucketing
        bucket = int((v - min_val) / range_val * num_buckets)
        bucket = min(bucket, num_buckets - 1)

        tokens.append("<NUM>")
        features.append({
            "value": v,
            "norm_value": norm_value,
            "log_value": log_value,
            "delta": delta,
            "norm_delta": norm_delta,
            "bucket": bucket
        })

        prev_value = v

    return tokens, features
