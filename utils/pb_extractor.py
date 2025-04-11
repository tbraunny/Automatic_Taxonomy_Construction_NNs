import numpy as np
import tensorflow as tf
import json
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import node_def_pb2

uselss_layers = [ # layers that do not provide us any benefit
    "DecodeJpeg",
    "Cast",
    "ExpandDims",
    "ResizeBilinear",
    "Sub",
    "Mul",
    "CheckNumerics",
    "Identity"
]

class PBExtractor:
    def extract_compute_graph(pb_file: str , output_json: str) -> None:
        """
        Extracts the computation graph from a tensorflow .pb file and saves it as JSON
        NOTE: useless layer types: DecodeJPEG, Cast, ExpandDims, ResizeBilinear, sub (tensor subtraction),
        mul (* of 2 tensors), CheckNumerics, identity, 

        :param pb_file: input tensorflow file
        :param output_json: output path for parsed JSON
        :return None
        """
        with tf.io.gfile.GFile(pb_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            try:
                graph_def.ParseFromString(f.read())
                print("Graph loaded successfully")
            except Exception as e:
                print(f"Error loading graph: {e}")

        consumer_map = {} # for target layers
        for node in graph_def.node:
            for input_name in node.input:
                clean_input = input_name.split(":")[0]
                if clean_input not in consumer_map:
                    consumer_map[clean_input] = []
                consumer_map[clean_input].append(node.name)

        const_nodes = {}
        minimal_graph: list = []
        for node in graph_def.node:  
            if node.op == "Const": # Const has become useful!
                try:
                    const_nodes[node.name] = tensor_util.MakeNdarray(node.attr["value"].tensor) # for parameter counting
                    continue # not useful in JSON
                except Exception:
                    continue # skip if tensor parsing fails
            if node.op in uselss_layers:
                continue
            
            target_layers: list = consumer_map.get(node.name , [])
            node_info: dict = {
                "name": node.name,
                "type": None, # for flagging in instantation (layers without a type)
                "target": target_layers,
                "input": list(node.input),
                "num_params": None
            }

            # like Chase's PyTorch-ONNX mapping, but for tensorflow to pytorch (INCOMPLETE)
            tf_to_pytorch_type_map: dict = {
                "Conv2D": "Conv2d",
                "Relu": "ReLU",
                "MaxPool": "MaxPool2d",
                "MatMul": "Linear",
                "Dense": "Linear",
                "BiasAdd": "Bias",
                "BatchNormWithGlobalNormalization": "batch_norm",
                "AvgPool": "AveragePooling2D",
                "Concat": "cat",
                "reduce_prod": "prod"
            }
            node_info["type"] = tf_to_pytorch_type_map.get(node.op, node.op)

            total_params = 0
            for input_name in node.input:
                clean_name = input_name.split(":")[0]
                if clean_name in const_nodes:
                    tensor = const_nodes[clean_name]
                    total_params += np.prod(tensor.shape) if hasattr(tensor, "shape") else 0

            node_info["num_params"] = int(total_params)

            minimal_graph.append(node_info)

        with open(output_json, "w") as json_file:
            json.dump({"network": minimal_graph}, json_file, indent=3)
        
        print(f"Graph extracted and saved to {output_json}")


if __name__ == "__main__":
    pb_file_path = "data/pb_testing/inception.pb" 
    PBExtractor.extract_compute_graph(pb_file_path, "data/pb_testing/pb_inception_architecture.json")