import numpy as np
import pandas as pd
import torch


def save_svals(svals_dict: dict, sigma: np.ndarray, block: str, dir: str = "data"):
    df = pd.DataFrame(svals_dict)
    sigmas = pd.DataFrame(sigma, columns=df.columns, index=["sigma"])
    df = pd.concat((sigmas, df))
    df.to_csv(f"{dir}/svals/{block}.csv")
    return

def save_overlaps(overlaps_dict: dict, block: str, dir: str = "data"):
    df = pd.DataFrame(overlaps_dict)
    df.to_csv(f"{dir}/overlaps/{block}.csv")
    return

def save_projections(projections_dict: dict, block: str, dir: str = "data"):
    df = pd.DataFrame(projections_dict)
    df.to_csv(f"{dir}/projections/{block}.csv")
    return