import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from graphviz import Digraph
import json 
from torchvision import models as tmodels

def extract_graph(model):
    traced = symbolic_trace(model)
    computeGraph = {"nodes": []}
    for node in traced.graph.nodes:
        node_info = {
            "op": node.op,
            "name": node.name,
            "target": str(node.target),
            "input": [str(arg) for arg in node.args],  # Convert inputs to strings for JSON serialization
            "parameters": {}  # Initialize empty parameters
        }

        # If the node is calling a module, retrieve its type and parameter shapes
        if node.op == 'call_module':
            module = traced.get_submodule(node.target)
            module_type = type(module).__name__
            node_info['type'] = module_type

            # Retrieve shapes of parameters
            param_shapes = {k: list(v.shape) for k, v in module.state_dict().items()}
            node_info['parameters'] = param_shapes

        elif node.op == 'call_function':
            node_info['type'] = f"Function: {str(node.target)}"
            # Get the name of the built-in function (like add, relu, etc.)
            if hasattr(node.target, '__name__'):
                func_name = node.target.__name__
            else:
                func_name = str(node.target)  # Fallback for complex functions
            node_info['type'] = f"{func_name}"
        computeGraph['nodes'].append(node_info)

    # Convert to JSON and return
    return json.dumps(computeGraph, indent=4)

if __name__ == "__main__":
    # Define a simple model with skip connection
    class SimpleSkipConnectionModel(nn.Module):
        def __init__(self):
            super(SimpleSkipConnectionModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        def forward(self, x):
            identity = x  # Skip connection branch
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out += identity  # Skip connection (add original input)
            return out

    # Step 1: Trace the model using torch.fx
    model = SimpleSkipConnectionModel()
    model = tmodels.get_model('resnet50',weights='DEFAULT')
    print(extract_graph(model))
