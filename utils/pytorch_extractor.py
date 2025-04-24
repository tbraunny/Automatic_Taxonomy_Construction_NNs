import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import json 
from torchvision import models as tmodels
from collections import defaultdict

def extract_graph(model) -> dict:
    traced = symbolic_trace(model)
    computeGraph = {
        "total_num_params": 0,
        "graph": {"node": []}
        }

    node_outputs: dict = defaultdict(list) # bypass init loop
    for node in traced.graph.nodes:
        for arg in node.args:
            if hasattr(arg, 'name'): # build target layers dict
                node_outputs[arg.name].append(node.name)

    total_num_params: int = 0

    for node in traced.graph.nodes: 
        node_info = {
            "name": node.name,
            "opType": None,
            "op": node.op,
            "output": node_outputs[node.name],#str(node.target),
            "input": list([str(arg) for arg in node.args]),  # Convert inputs to strings for JSON serialization
            "attributes": {},  # Initialize empty parameters
            "num_params": None # important stuff
        }

        # If the node is calling a module, retrieve its type and parameter shapes
        if node.op == 'call_module':
            module = traced.get_submodule(node.target)
            module_type = type(module).__name__
            node_info['op_type'] = module_type

            # Retrieve shapes of parameters
            param_shapes = {k: list(v.shape) for k, v in module.state_dict().items()}
            node_info['attributes'] = param_shapes

            # Retrieve total number of learned parameters
            param_count: int = 0
            submodule = traced.get_submodule(node.target)
            param_count = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            node_info["num_params"] = param_count

            # Retrieve total number of learned parameters
            param_count: int = 0
            submodule = traced.get_submodule(node.target)
            param_count = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            total_num_params += param_count
            node_info["num_params"] = param_count

        elif node.op == 'call_function':
            node_info['op_type'] = f"Function: {str(node.target)}"
            # Get the name of the built-in function (like add, relu, etc.)
            if hasattr(node.target, '__name__'):
                func_name = node.target.__name__
            else:
                func_name = str(node.target)  # Fallback for complex functions
            node_info['op_type'] = f"{func_name}"
        computeGraph['graph']['node'].append(node_info)

    computeGraph['total_num_params'] = total_num_params
    return computeGraph
    #return json.dumps(computeGraph, indent=4)

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
    symbolic_graph_dict = extract_graph(model)

    with open("data/pytorch_test.json" , "w") as f:
        json.dump(symbolic_graph_dict , f , indent=2)

    print("PyTorch parsed!")