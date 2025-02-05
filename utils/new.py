instantiation_map = {
    # AnnConfig hasNetwork
        "Network" : ["Network"],
        "Layer" : ["Conv",
                    "InputLayer",
                    "OutputLayer",
                    "Elu",
                    "Relu",
                    "Relu6",
                    "Sigmoid",
                    "SoftMax",
                    "SoftPlus ",
                    "SoftSign ",
                    "Tanh ",
                    "GRULayer ",
                    "LSTMLayer ",
                    "ConvolutionLayer",
                    "DeconvolutionLayer",
                    "FullyConnectedLayer",
                    "BidirectionalRNNLayer",
                    "ConcatLayer",
                    "MultiplyLayer",
                    "PoolingLayer",
                    "SumLayer",
                    "UpscaleLayer",
                    "Exponential",
                    "Gaussian",
                    "Categorical",
                    "Normal",
                    "Uniform",
                    "DropoutLayer",
                    "BatchNormLayer",
                    "FlattenLayer",
                    "CloneLayer",
                    "SplitLayer"],
        "ObjectiveFunction" : [
                    "MinObjectiveFunction",
                    "MaxObjectiveFunction",


                - Connected Class: CostFunction [FIR]
                        - Connected Class: LossFunction<IS>
                                - Subclass: BinaryCrossEntropy [FIR]
                                - Subclass: CategoricalCrossEntropy [FIR]
                                - Subclass: MSE [FIR]
                        - Connected Class: RegularizerFunction<IS>
                                - Subclass: L1L2RegularizerFunction [FIR]
                                - Subclass: L1RegularizerFunction [FIR]
                                - Subclass: L2RegularizerFunction [FIR]
                - Subclass: MaxObjectiveFunction [FIR]
                - Subclass: MinObjectiveFunction [FIR]
        "TaskCharacterization" : [
                 Adversarial [FIR]
                         SemiSupervisedClassification [FIR]
                         SupervisedClassification [FIR]
                 Clustering [FIR]
                 Discrimination [FIR]
                 Generation [FIR]
                 Reconstruction [FIR]
                 Regression [FIR]
        ]
        ]

    
}




layer_list = ["conv layer", "conv layer", "conv layer", "deconv layer", "outputlayer"]


for i, layer_name enumerate()
network.hasLayer = instantiation (name)