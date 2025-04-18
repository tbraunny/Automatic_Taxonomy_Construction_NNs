import torch
import torch.nn as nn
import json

"""
NOTE: DEPRECATED
"""

class PTExtractor:
    def extract_compute_graph(pt_file: str , output_json: str) -> None:
        """
        Extracts computation graph from a PyTorch weights & biases file (*.pt) & parses it
        into a JSON

        :param pt_file: input pytorch file
        :param output_json: output path for parsed JSON
        :return None
        """
        # Load the PyTorch model
        model = torch.load(pt_file)

        if isinstance(model , dict):
            model = 0 # nowhere to go from here

        model.eval()

        model_info = {
            "layers": []
        }

        # Iterate over the named layers in the model
        for name, module in model.named_modules():
            layer_info = {
                "name": name,
                "type": str(type(module)),
                "parameters": {},
                "input": [],
                "output": []
            }

            # Extract parameters for layers with weights and biases
            if isinstance(module, nn.Conv2d):
                layer_info["type"] = "Conv2d"
                if module.weight is not None:
                    layer_info["parameters"]["weight"] = module.weight.shape
                if module.bias is not None:
                    layer_info["parameters"]["bias"] = module.bias.shape
                layer_info["parameters"]["kernel_size"] = module.kernel_size
                layer_info["parameters"]["stride"] = module.stride
                layer_info["parameters"]["padding"] = module.padding

            elif isinstance(module, nn.Linear):
                layer_info["type"] = "Linear"
                if module.weight is not None:
                    layer_info["parameters"]["weight"] = module.weight.shape
                if module.bias is not None:
                    layer_info["parameters"]["bias"] = module.bias.shape

            elif isinstance(module, nn.BatchNorm2d):
                layer_info["type"] = "BatchNorm2d"
                if module.weight is not None:
                    layer_info["parameters"]["weight"] = module.weight.shape
                if module.bias is not None:
                    layer_info["parameters"]["bias"] = module.bias.shape

            elif isinstance(module, nn.LSTM):
                layer_info["type"] = "LSTM"
                # Extracting LSTM-specific parameters
                layer_info["parameters"]["input_size"] = module.input_size
                layer_info["parameters"]["hidden_size"] = module.hidden_size
                layer_info["parameters"]["num_layers"] = module.num_layers

            elif isinstance(module, nn.RNN):
                layer_info["type"] = "RNN"
                # Extracting RNN-specific parameters
                layer_info["parameters"]["input_size"] = module.input_size
                layer_info["parameters"]["hidden_size"] = module.hidden_size
                layer_info["parameters"]["num_layers"] = module.num_layers

            # Add more checks for other layer types if needed (e.g., dropout, activation)

            # Add the extracted layer info to the model info
            model_info["layers"].append(layer_info)
        
        with open(output_json, "w") as json_file:
            json.dump(model_info , json_file, indent=3)
        
        print(f"Graph extracted and saved to {output_json}")

if __name__ == '__main__':
    pt_file_path = "data/pt_testing/planesnet_weights.pt"
    PTExtractor.extract_compute_graph(pt_file_path , "data/pt_testing/planesnet_parsed.json")