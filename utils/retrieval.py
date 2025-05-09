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