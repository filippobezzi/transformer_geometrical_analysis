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
        "ln_1": [],
        "attn.c_attn": [],
        "attn.c_proj": [],
        "attn": [],
        "ln_2": [],
        "mlp.c_fc": [],
        "mlp.act": [],
        "mlp.c_proj": [],
        "mlp": []
    }

    def hook_ln_1(module, input, output):
        activations["ln_1"].append(output[0])

    def hook_attn_c_attn(module, input, output):
        activations["attn.c_attn"].append(output[0])

    def hook_attn_c_proj(module, input, output):
        activations["attn.c_proj"].append(output[0])
    
    def hook_attn(module, input, output):
        activations["attn"].append(output[0][0])

    def hook_ln_2(module, input, output):
        activations["ln_2"].append(output[0])
    
    def hook_mlp_c_fc(module, input, output):
        activations["mlp.c_fc"].append(output[0])

    def hook_mlp_act(module, input, output):
        activations["mlp.act"].append(output[0])

    def hook_mlp_c_proj(module, input, output):
        activations["mlp.c_proj"].append(output[0])

    def hook_mlp(module, input, output):
        activations["mlp"].append(output[0])

    # we now register the hooks so that at each block they are called
    for i, block in enumerate(model.transformer.h):
        block.ln_1.register_forward_hook(hook_ln_1)
        block.attn.c_attn.register_forward_hook(hook_attn_c_attn)
        block.attn.c_proj.register_forward_hook(hook_attn_c_proj)
        block.attn.register_forward_hook(hook_attn)
        block.ln_2.register_forward_hook(hook_ln_2)
        block.mlp.c_fc.register_forward_hook(hook_mlp_c_fc)
        block.mlp.act.register_forward_hook(hook_mlp_act)
        block.mlp.c_proj.register_forward_hook(hook_mlp_c_proj)
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