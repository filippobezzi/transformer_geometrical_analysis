import torch

def get_activations(model):
    """
    Function that attaches hooks that save the activations of the model
    Args:
        model (GPT2LMHeadModel): the gpt2 model in which hook functions will be attached
    Returns:
        activations (dict): empty dictionary where the activations of attention and mlp will be stored 
    """

    # dict to store all activations
    activations = {
        "attn_inputs": [],
        "attn_attentions": [], # this will have to be saved manually from output.activations
        "attn_outputs": [],
        "mlp_inputs": [],
        "mlp_outputs": []
    }

    # hook functions to save the activations
    def hook_attention(module, input, output):
        # input: (hidden_states,) | output: hidden_states after attention
        activations["attn_inputs"].append(input[0][0])
        # we save the embedded vectors as a whole and not partitioned as they have been processed by the heads
        activations["attn_outputs"].append(output[0][0]) 

    # same for the mlp
    def hook_mlp(module, input, output):
        activations["mlp_inputs"].append(input[0][0])
        activations["mlp_outputs"].append(output[0])

    # we now register the hooks so that at each block they are called
    for i, block in enumerate(model.transformer.h):
        block.attn.register_forward_hook(hook_attention)
        block.mlp.register_forward_hook(hook_mlp)
    
    return activations

def get_all_embedded_vectors(model):
    """
    Args:
        model (GPT2LMHeadModel): the gpt2 model
    Returns:
        vector_embedding (torch.Tensor): tensor containing the vectors of each possible token
    """

    # self explanatory constants
    MAX_TOKEN_INPUT_SIZE = 1024
    TOTAL_VOCAB_SIZE = 50256
    EMBEDDING_DIMENTIONS = 768

    # defining a tensor of 0s that will later be removed
    vector_embedding = torch.tensor([0]*EMBEDDING_DIMENTIONS).reshape(1, -1)

    for i in range(0, TOTAL_VOCAB_SIZE, MAX_TOKEN_INPUT_SIZE):
        # ranging thru every possible token in gpt2 vocabulary
        # 1024 at a time cause that's the max it takes
        input_ids = torch.arange(i, min(i+MAX_TOKEN_INPUT_SIZE, TOTAL_VOCAB_SIZE + 1)).reshape(1, -1)
        with torch.no_grad():
            vector_embedding = torch.cat((vector_embedding, model.transformer.wte(input_ids)[0]), dim = 0)

    return vector_embedding[1:,:]
