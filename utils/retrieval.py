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
        "wte_outputs": [],
        "wpe_outputs": [],
        "ln_1_inputs": [],
        "ln_1_outputs": [],
        # "attn_inputs": [], # same as ln_1_outputs
        "attn_attentions": [], # this will have to be saved manually from output.activations
        "attn_outputs": [],
        "ln_2_inputs": [],
        "ln_2_outputs": [],
        # "mlp_inputs": [], # same as ln_2_outputs
        "mlp_outputs": []
    }

    def hook_wte(module, input, output):
        activations["wte_outputs"].append(output[0])

    def hook_wpe(module, input, output):
        activations["wpe_outputs"].append(output[0])

    # hook functions to save the activations
    def hook_attention(module, input, output):
        # input: (hidden_states,) | output: hidden_states after attention
        # activations["attn_inputs"].append(input[0][0])
        # we save the embedded vectors as a whole and not partitioned as they have been processed by the heads
        activations["attn_outputs"].append(output[0][0]) 

    # same for the mlp
    def hook_mlp(module, input, output):
        # activations["mlp_inputs"].append(input[0][0])
        activations["mlp_outputs"].append(output[0])

    # same for the ln_1
    def hook_ln_1(module, input, output):
        activations["ln_1_inputs"].append(input[0][0])
        activations["ln_1_outputs"].append(output[0])

    # same for the ln_2
    def hook_ln_2(module, input, output):
        activations["ln_2_inputs"].append(input[0][0])
        activations["ln_2_outputs"].append(output[0])

    model.transformer.wte.register_forward_hook(hook_wte)
    model.transformer.wpe.register_forward_hook(hook_wpe)
    # we now register the hooks so that at each block they are called
    for i, block in enumerate(model.transformer.h):
        block.ln_1.register_forward_hook(hook_ln_1)
        block.attn.register_forward_hook(hook_attention)
        block.ln_2.register_forward_hook(hook_ln_2)
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
    TOTAL_VOCAB_SIZE = 50257
    EMBEDDING_DIMENTIONS = 768

    # defining a tensor of 0s that will later be removed
    vector_embedding = torch.tensor([0]*EMBEDDING_DIMENTIONS).reshape(1, -1)

    for i in range(0, TOTAL_VOCAB_SIZE, MAX_TOKEN_INPUT_SIZE):
        # ranging thru every possible token in gpt2 vocabulary
        # 1024 at a time cause that's the max it takes
        input_ids = torch.arange(i, min(i+MAX_TOKEN_INPUT_SIZE, TOTAL_VOCAB_SIZE)).reshape(1, -1)
        with torch.no_grad():
            vector_embedding = torch.cat((vector_embedding, model.transformer.wte(input_ids)[0]), dim = 0)

    return vector_embedding[1:,:]
