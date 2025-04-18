def check_actfunc() -> list:
    """
    Check if the given layer is a known layer type (activation, pooling, normalization)

    :return known_activation_functions: List of known activation functions
    """

    known_activation_functions = [
        "Softmax",
        "ReLU",
        "Tanh",
        "Linear",
        "GAN_Generator_Tanh",
        "GAN_Generator_ReLU",
        "GAN_Discriminator_Sigmoid",
        "GAN_Discriminator_ReLU",
        "AAE_Encoder_Linear",
        "AAE_Encoder_ReLU",
        "AAE_Encoder_Softmax",
        "AAE_Encoder_ZClone_ReLU",
        "AAE_Encoder_YClone_ReLU",
        "AAE_Decoder_Sigmoid",
        "AAE_Decoder_ReLU",
        "AAE_Style_Discriminator_ReLU",
        "AAE_Style_Discriminator_Sigmoid",
        "AAE_Label_Discriminator_Sigmoid",
        "AAE_Label_Discriminator_ReLU",
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
        "AdaptiveLogSoftmaxWithLoss"
    ]

    return known_activation_functions
    
def check_pooling() -> list:
    """
    Check if the given layer is a known pooling layer

    "return: known_pooling_layers: List of known pooling layers
    """
    known_pooling_layers = [
        "MaxPool",
        "MaxPool1d"
        "MaxPool2d",
        "MaxPool3d",
        "AvgPool",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "FractionalMaxPool2d",
        "FractionalMaxPool3d",
        "LPpool2d",
        "LPPool3d",
        "AdaptiveMaxPool1d",
        "AdaptiveMaxPool2d",
        "AdaptiveMaxPool3d",
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
        "GlobalPool",
        "GlobalPool1d",
        "GlobalPool2d",
        "GlobalPool3d"
    ]

    return known_pooling_layers

def check_norm() -> list:
    """
    Check if the given layer is a known normalization layer

    :return known_norm_layers: List of known normalization layers
    """
    known_norm_layers = [
        "BatchNormalization",
        "BatchNorm",
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "LazyBatchNorm",
        "LazyBatchNorm1d",
        "LazyBatchNorm2d",
        "LazyBatchNorm3d",
        "GroupNorm",
        "SyncBatchNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        "LazyInstanceNorm1d",
        "LazyInstanceNorm2d",
        "LazyInstanceNorm3d",
        "LayerNorm"
        "LocalResponseNorm"
        "RMSNorm"
    ]

    return known_norm_layers