import numpy as np
import pandas as pd
import os

from utils.constants import *

def save_svals(svals_dict: dict, sigma: np.ndarray, layer: str, dir: str = DATA_DIR):
    """
    Function that saves a file "dir/svals/layer.csv", 
    where each column corresponds to a block, 
    the first row is dedicated to the sigma values and
    every other row contains the singular values of the corresponding weight matrices.
    Args:
        svals_dict (dict): each key is a different block, and the corresponding value contains the svals of the weight matrix
        sigma (np.ndarray): standard deviation of the weight matrix. Can be useful to compute Marchenko-Pastur parameters
        layer (str): name of the (sub-)layer
        dir (str, default: DATA_DIR): base directory where the file is saved
    """
    df = pd.DataFrame(svals_dict)
    sigmas = pd.DataFrame(sigma, columns=df.columns, index=["sigma"])
    df = pd.concat((sigmas, df))
    saving_dir = f"{dir}/svals"
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir) 
    df.to_csv(f"{saving_dir}/{layer}.csv")
    return

def save_overlaps(overlaps_dict: dict, layer: str, dir: str = DATA_DIR):
    """
    Function that saves a file "dir/overlaps/layer.csv", 
    where each column corresponds to a block and 
    every row contains the overlap values.
    Args:
        overlaps_dict (dict): each key is a different block, and the corresponding value contains the overlaps
        layer (str): name of the (sub-)layer
        dir (str, default: DATA_DIR): base directory where the file is saved
    """
    df = pd.DataFrame(overlaps_dict)
    saving_dir = f"{dir}/overlaps"
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir) 
    df.to_csv(f"{saving_dir}/{layer}.csv")
    return