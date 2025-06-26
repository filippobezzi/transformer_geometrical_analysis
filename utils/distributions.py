import numpy as np
import torch

def marchenko_pastur_svals(x: torch.tensor, sigma: float, m: int, n: int):
    """
    Marchenko-Pastur distribution for singular values
    
    Args:
        x (torch.tensor):
        sigma (float):
        m (int):
        n (int):
    
    Returns:
        tuple (torch.tensor, float, float): values of the distribution for `x`, s_lower and s_upper
    """
    q = m / n
    s_lower = sigma * np.fabs(np.sqrt(m) - np.sqrt(n))
    s_upper = sigma * np.fabs(np.sqrt(m) + np.sqrt(n))

    def _dist(x):
        return torch.sqrt((s_upper**2 - x**2) * (x**2 - s_lower**2)) / (torch.pi * sigma**2 * q * x)
    
    return torch.where(
        torch.logical_and(x > s_lower, x < s_upper),
        _dist(x),
        0
    ), s_lower, s_upper