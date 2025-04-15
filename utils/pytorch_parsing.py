import torch.nn as nn

def is_activation_layer(layer_name: str, use_torch: bool=False) -> bool:
    """Check if the layer is an activation function. (Non Case-Sensitive)
    
    Args:
        layer_name (str): Name of the layer.
        use_torch (bool): If True, check against PyTorch activation functions.
    """
    layer_name = layer_name.lower()

    if use_torch:
        layer_class = getattr(nn, layer_name, None)
        return 'activation' in str(layer_class).lower() if layer_class else False
        
    activation_layers = [
        # Non-Linear activations (weighted sum, nonlinearity)
        "ELU",
        "Hardshrink",
        "Hardsigmoid",
        "Hardtanh",
        "Hardswish",
        "LeakyReLU",
        "LogSigmoid",
        "MultiheadAttention",
        "PReLU",
        "ReLU",
        "ReLU6",
        "RReLU",
        "SELU",
        "CELU",
        "GELU",
        "Sigmoid",
        "SiLU",
        "Mish",
        "Softplus",
        "Softshrink",
        "Softsign",
        "Tanh",
        "Tanhshrink",
        "Threshold",
        "GLU",
        # Non-Linear activations (others)
        "Softmin",
        "Softmax",
        "Softmax2d",
        "LogSoftmax",
        "AdaptiveLogSoftmaxWithLoss",
    ]
    return any(layer_name.startswith(act.lower()) for act in activation_layers)