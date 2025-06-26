import pandas as pd
import torch


def save_svals(svals_dict: dict, block: str):
    df = pd.DataFrame(svals_dict)
    df.to_csv(f"data/svals/{block}.csv")
    return

def save_overlaps(overlaps_dict: dict, block: str):
    df = pd.DataFrame(overlaps_dict)
    df.to_csv(f"data/overlaps/{block}.csv")
    return