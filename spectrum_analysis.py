import numpy as np
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.retrieval import get_activations, extract_weights
from utils.storage import save_overlaps, save_svals
from utils.visualization import plot_layer_overlaps, plot_layer_svals

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


def main():

    model, tokenizer = load_model()
    TOTAL_VOCAB_SIZE = tokenizer.vocab_size

    ### PROMPT CREATION/SELECTION
    n_prompts = 100
    n_tokens = 100
    prompt_type = "random"
    data_dir = None
    plot_dir = None

    # prompts are randomly generated
    if prompt_type == "random":
        data_dir = "prova/data"
        plot_dir = "prova/plots"
        prompts = [torch.randint(0, TOTAL_VOCAB_SIZE, size = (1, n_tokens)) for _ in range(n_prompts)]

    # prompts are taken from _input_phrases.txt
    elif prompt_type == "input_phrases":
        data_dir = "ip/data"
        plot_dir = "ip/plots"
        with open("_input_phrases.txt", "r") as f:
            phrases = [line.strip() for line in f.readlines() if line.strip()]

        prompts = []
        min_length = MAXIMUM_INPUT_LENGTH
        for p in phrases:
            i = tokenizer(p, return_tensors="pt")["input_ids"]
            l = i.shape[1]
            if l < min_length:
                min_length = l
            prompts.append(i)

        n_prompts = len(prompts)
        n_tokens = min_length

    data_dir = "results/data"
    plot_dir = "results/plots"

    ###

    input_activations = get_activations(model)

    ###

    # runnning all the prompts to obtain all activations
    for prompt in prompts:
        with torch.no_grad():
            # making sure all prompts have the same lenght
            _ = model(prompt[:,:n_tokens])

    ###

    # extracting all the weights from the transformer
    weights = extract_weights(model)

    ###

    sublayer_selection = input_activations.keys()


    def reorder_sublayer_acts(sublayer: str, activations: dict):
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


    for sublayer in sublayer_selection:
        # excluding the normalization layers from the analysis
        if sublayer in ["ln_1", "ln_2"]: continue

        print(sublayer)
        acts = reorder_sublayer_acts(sublayer, input_activations)

        w_sigmas = []
        svals_dict = {}
        overlaps_dict = {}
        projections_dict = {}
        for idx, X in acts.items():
            W = torch.transpose(weights[f"h.{idx}.{sublayer}.weight"], 0, 1)
            _, N, D = X.shape

            ###

            # computing the activation covariance matrix as defined in the paper (i.e. as the mean of covariances of the input buffers)
            x_mean = torch.mean(X, dim = [0,1])
            x_centered = X - x_mean
            act_cov = torch.sum(torch.matmul(torch.transpose(x_centered, 1, 2), x_centered), dim = 0) / (n_prompts * (N - 1))

            act_cov_eigvals, act_cov_eigvecs = torch.linalg.eigh(act_cov)

            # inverting the ordering (now descending)
            act_cov_eigvals = act_cov_eigvals.flip(0)
            act_cov_eigvecs = act_cov_eigvecs.flip(1)

            # computing the singular values of the centered weight matrix
            w_mean = torch.mean(W, dim = 0)
            w_centered = W - w_mean

            _, w_svals, w_Vt = torch.linalg.svd(w_centered)

            max_rank = np.min(W.shape)

            # removing all eigenvalues we are sure to be 0
            w_svals = w_svals[:max_rank]
            w_Vt = w_Vt[:max_rank,:]

            # and the corresponding eigenvectors in the act_cov
            act_cov_eigvals = act_cov_eigvals[:max_rank]
            act_cov_eigvecs = act_cov_eigvecs[:,:max_rank]

            # computing the overlap between w and act_cov sigular vectors
            projections = torch.matmul(w_Vt, act_cov_eigvecs)
            overlap = torch.max(torch.abs(projections), dim = 0)[0]


            ###

            # marchenko-pastur parameters
            n, m = W.shape
            sigma = torch.std(w_centered)


            w_sigmas.append(sigma.detach().numpy())
            svals_dict[f"{idx}"] = w_svals.detach().numpy()
            overlaps_dict[f"{idx}"] = overlap.detach().numpy()
            projections_dict[f"{idx}"] = projections.detach().numpy().reshape(-1,)


        save_svals(svals_dict, np.array(w_sigmas).reshape(1,-1), sublayer, data_dir)
        save_overlaps(overlaps_dict, sublayer, data_dir)
        # save_projections(projections_dict, sublayer, data_dir)
        plot_layer_svals(sublayer, m, n, data_dir, plot_dir)
        plot_layer_overlaps(sublayer, m, n, data_dir, plot_dir)
        # plot_layer_projections(sublayer, m, n, data_dir, plot_dir)
    return

if __name__=="__main__":
    main()
