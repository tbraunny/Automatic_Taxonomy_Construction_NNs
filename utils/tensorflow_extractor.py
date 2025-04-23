import json
import ast
import tensorflow as tf
from collections import defaultdict

def extract_tf_graph(class_node: ast.ClassDef) -> dict:
    """
    Extract & parse tensorflow code from .py files

    :param class_node: Class node
    :return symbolic_graph: Symbolic graph of model architecture
    """
    compute_graph = {"graph": {"node": []}}
    layer_defs = {}
    layer_calls = []

    # Step 1: Look for self.<name> = tf.keras.layers.<LayerType>() assignments
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == '__init__':
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.targets[0], ast.Attribute):
                        target = stmt.targets[0]
                        if (isinstance(stmt.value, ast.Call) and
                            isinstance(stmt.value.func, ast.Attribute)):
                            layer_name = target.attr
                            layer_type = stmt.value.func.attr
                            layer_defs[layer_name] = layer_type

    # Step 2: Look for self.<name>(...) calls inside the `call` method
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'call':
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                    if isinstance(stmt.func.value, ast.Name) and stmt.func.value.id == 'self':
                        layer_calls.append(stmt.func.attr)

    # Step 3: Build the graph
    for i, name in enumerate(layer_calls):
        compute_graph['graph']['node'].append({
            "name": name,
            "opType": layer_defs.get(name, "Unknown"),
            "op": "Layer",
            "input": [layer_calls[i-1]] if i > 0 else [],
            "output": [layer_calls[i+1]] if i < len(layer_calls) - 1 else [],
            "attributes": {},  # Optional: could parse args from init call
            "num_params": None
        })

    return compute_graph