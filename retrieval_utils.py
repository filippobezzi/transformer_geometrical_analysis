def get_activations(model):

    # dict to store all activations
    #   both from attention and from mlp
    activations = {
        "attn_inputs": [],
        "attn_outputs": [],
        "mlp_inputs": [],
        "mlp_outputs": []
    }

    # Hook functions to save the activations
    def hook_attention(module, input, output):
        # input: (hidden_states,) | output: hidden_states after attention
        activations["attn_inputs"].append(input)
        # we save the embedded vectors as a whole and not partitioned as they have been processed by the heads
        activations["attn_outputs"].append(output[0]) 

    def hook_mlp(module, input, output):
        activations["mlp_inputs"].append(input)
        activations["mlp_outputs"].append(output)

    # we now register the hooks so that at each block they are called
    for i, block in enumerate(model.transformer.h):
        block.attn.register_forward_hook(hook_attention)
        block.mlp.register_forward_hook(hook_mlp)
    
    return activations