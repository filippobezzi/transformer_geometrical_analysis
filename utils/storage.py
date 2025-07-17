import numpy as np
import pandas as pd
import os
import torch


def save_svals(svals_dict: dict, sigma: np.ndarray, block: str, dir: str = "data"):
    df = pd.DataFrame(svals_dict)
    sigmas = pd.DataFrame(sigma, columns=df.columns, index=["sigma"])
    df = pd.concat((sigmas, df))
    saving_dir = f"{dir}/svals"
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir) 
    df.to_csv(f"{saving_dir}/{block}.csv")
    return

def save_overlaps(overlaps_dict: dict, block: str, dir: str = "data"):
    df = pd.DataFrame(overlaps_dict)
    saving_dir = f"{dir}/overlaps"
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir) 
    df.to_csv(f"{saving_dir}/{block}.csv")
    return

def save_projections(projections_dict: dict, block: str, dir: str = "data"):
    df = pd.DataFrame(projections_dict)
    saving_dir = f"{dir}/projections"
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir) 
    df.to_csv(f"{saving_dir}/{block}.csv")
    return