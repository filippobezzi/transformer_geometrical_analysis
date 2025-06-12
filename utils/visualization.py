import numpy as np
import torch
import matplotlib.pyplot as plt

def mp_weights(weights: dict, layer: str):

    fig, axs = plt.subplots(ncols=12, figsize=(3*12,3))
    for i in range(12):
        w = weights[f"h.{i}.{layer}.weight"]
        w_mean = torch.mean(w, dim = 0)
        w_centered = w - w_mean
        _, s, _ = torch.linalg.svd(w_centered)
        # axs[i].hist(s, bins = "fd", density = True)

        w_sigma = torch.std(w_centered).detach().numpy()
        m, n = w.shape
        x = np.linspace(0, np.max(s.detach().numpy()), 1000)
        axs[i].plot(x, mp_distribution(x, w_sigma, m, n))
    
    fig.tight_layout()
    plt.show()
    return

def mp_distribution(x, sigma, m, n):
    var_ = sigma**2 / np.sqrt(n)
    nu_plus = var_ * (1 + np.sqrt(m / n))
    nu_minus = var_ * (1 - np.sqrt(m / n))
    return np.where(np.logical_and(x < nu_plus, x > nu_minus), n / (np.pi * m * x * var_) * np.sqrt((nu_plus**2 - x**2) * (x**2 - nu_minus**2)), 0)