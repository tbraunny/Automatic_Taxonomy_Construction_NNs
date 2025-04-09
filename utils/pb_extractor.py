import tensorflow as tf
import json

class PBExtractor:
    def extract_compute_graph(pb_file: str , output_json: str):
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

        minimal_graph: list = []
        for node in graph_def.node:
            if node.op == "Const":  # ignore const values
                continue  

            node_info: dict = {
                "name": node.op,
                "type": None, # for flagging in instantation (layers without a type)
                "target": node.name,
                "input": list(node.input),
                "parameters": {} # for later (very much needed)
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
            minimal_graph.append(node_info)

        with open(output_json, "w") as json_file:
            json.dump({"network": minimal_graph}, json_file, indent=3)
        
        print(f"Graph extracted and saved to {output_json}")


if __name__ == "__main__":
    pb_file_path = "data/pb_testing/inception.pb" 
    PBExtractor.extract_compute_graph(pb_file_path, "data/pb_testing/pb_inception_architecture.json")