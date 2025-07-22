import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.retrieval import get_activations, extract_weights
from utils.storage import save_overlaps, save_svals
from utils.visualization import plot_overlap_bands, plot_overlap_comparison_bands

# constants of the model
EMBEDDING_DIMENTIONS = 768
MAXIMUM_INPUT_LENGTH = 1024
N_BLOCKS = 12

def load_model(gpt2_version: str = None):

    if gpt2_version is None:
        gpt2_version = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)
    model = GPT2LMHeadModel.from_pretrained(gpt2_version, output_hidden_states=True, return_dict_in_generate=True)
    return model, tokenizer

def create_prompts(n_prompts: int, n_tokens: int, prompt_type: str, tokenizer):
    TOTAL_VOCAB_SIZE = tokenizer.vocab_size
    # prompts are randomly generated
    if prompt_type == "random":
        with torch.no_grad():
            prompts = [torch.randint(0, TOTAL_VOCAB_SIZE, size = (1, n_tokens)) for _ in range(n_prompts)]
    
    # we take as prompts `n_prompts` random `n_tokens`-long passages of an umberto eco essay
    elif prompt_type == "eco":
        with open("eco.txt", "r") as f:
            phrases = [line.strip() for line in f.readlines() if line.strip()]
        with torch.no_grad():
            tokens = tokenizer(phrases[0], return_tensors="pt")["input_ids"]
        prompts = []
        for i in range(n_prompts):
            idx = np.random.randint(0, tokens.shape[1] - n_tokens)
            prompts.append(tokens[:,idx:(idx + n_tokens)])
    
    # we take as prompts `n_prompts` random `n_tokens`-long passages of dune
    elif prompt_type == "dune":
        with open("dune.txt", "r") as f:
            phrases = [line.strip() for line in f.readlines() if line.strip()]
        with torch.no_grad():
            tokens = tokenizer(phrases[0], return_tensors="pt")["input_ids"]
        prompts = []
        for i in range(n_prompts):
            idx = np.random.randint(0, tokens.shape[1] - n_tokens)
            prompts.append(tokens[:,idx:(idx + n_tokens)])

    return prompts

def reorder_sublayer_acts(sublayer: str, activations: dict, n_prompts: int):
    """
    Given a dict `activations` obtained from `get_activations`,
    it selects only elements relative to `sublayer` and
    separates them in the 12 blocks, maintaining the prompt order.
    Args:
        sublayer (str):
        activations (dict):
    Returns:
        ordered_acts (dict):
    """
    acts = activations[sublayer]
    # selecting an element since the value is a list
    N, D = acts[0].shape
    ordered_acts = {
        # keys are the block number
        str(i):
            # the value is a torch.tensor of shape (n_prompts, n, d)
            torch.cat(
                [acts[p * N_BLOCKS + i].reshape(1,N,D) for p in range(n_prompts)],
                dim = 0
            )
        for i in range(N_BLOCKS)
    }
    return ordered_acts

def main(sublayer: str = None, block: int = None, prompt_type: str = None):
    # load model
    model, tokenizer = load_model()

    # initialize activations
    input_activations = get_activations(model)

    # save weights
    weights = extract_weights(model)

    # compute weight svals
    W = torch.transpose(weights[f"h.{block}.{sublayer}.weight"], 0, 1)

    w_mean = torch.mean(W, dim = 0)
    w_centered = W - w_mean

    _, w_svals, w_Vt = torch.linalg.svd(w_centered)

    max_rank = np.min(W.shape)

    # removing all eigenvalues we are sure to be 0
    w_svals = w_svals[:max_rank]
    w_Vt = w_Vt[:max_rank,:]

    n_iter = 10
    n_prompts = 10
    n_tokens = 100
    
    overlaps_dict = {}
    # iterate:
    for iter in range(n_iter):
        # generate prompts
        prompts = create_prompts(n_prompts, n_tokens, prompt_type, tokenizer)

        # run prompts
        for prompt in prompts:
            with torch.no_grad():
                _ = model(prompt)

        # compute acm and eigvecs
        acts = reorder_sublayer_acts(sublayer, input_activations, n_prompts)
        X = acts[f"{block}"]
        _, N, D = X.shape

        # computing the activation covariance matrix as defined in the paper (i.e. as the mean of covariances of the input buffers)
        x_mean = torch.mean(X, dim = [0,1])
        x_centered = X - x_mean
        act_cov = torch.sum(torch.matmul(torch.transpose(x_centered, 1, 2), x_centered), dim = 0) / (n_prompts * (N - 1))

        act_cov_eigvals, act_cov_eigvecs = torch.linalg.eigh(act_cov)

        # inverting the ordering (now descending)
        act_cov_eigvals = act_cov_eigvals.flip(0)
        act_cov_eigvecs = act_cov_eigvecs.flip(1)

        act_cov_eigvals = act_cov_eigvals[:max_rank]
        act_cov_eigvecs = act_cov_eigvecs[:,:max_rank]

        # compute overlap
        projections = torch.matmul(w_Vt, act_cov_eigvecs)
        overlap = torch.max(torch.abs(projections), dim = 0)[0]

        # save overlap in dict
        overlaps_dict[f"{iter}"] = overlap.detach().numpy()

        # reset activation dict
        for key in input_activations.keys():
            input_activations[key] = []
    
    # save overlap dict into csv file
    df = pd.DataFrame(overlaps_dict)
    saving_dir = f"final/data/overlaps/{prompt_type}/{sublayer}"
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir) 
    df.to_csv(f"{saving_dir}/{block}.csv")

    return

sublayers = [
    "attn.c_attn.q",
    "attn.c_attn.k",
    "attn.c_attn.v",
    "mlp.c_fc",
    "mlp.c_proj"
]

blocks = range(12)
N = [768, 768, 768, 4 * 768, 4 * 768]
data_dir = "final/data"
plot_dir = "final/plots"

if __name__=="__main__":
    with torch.no_grad():
        for sublayer, n in zip(sublayers, N):
            for block in blocks:
                print(sublayer, block)

                # main(sublayer, block, "random")
                # main(sublayer, block, "eco")
                
                plot_overlap_bands(sublayer, block, "random", 768, n, data_dir, plot_dir)
                plot_overlap_bands(sublayer, block, "eco", 768, n, data_dir, plot_dir)
                plot_overlap_comparison_bands(sublayer, block, 768, n, data_dir, plot_dir)
                
            break
