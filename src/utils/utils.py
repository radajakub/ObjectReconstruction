def is_in_range_approx(val: float, min: float, max: float, eps: float = 1e-6) -> bool:
    return (min - eps) <= val <= (max + eps)
