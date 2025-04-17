import json
from owlready2 import get_ontology
from inferenceutils import loadModel,load_dataset,get_embedding_dataset 

def extract_layer_parameters(layer):
    """
    Return a dictionary of data property values for a given layer.
    This example considers a fixed set of property names that might be 
    used as parameters. Adjust this mapping to suit your ontology.
    """
    # List the property names that we consider "parameters"
    param_props = ["axes", "value", "dilations", "group", "kernel_shape", "pads", "strides", "ceil_mode", "layer_num_units"]
    params = {}
    
    # Loop through all data properties on the layer
    for prop in layer.get_properties():
        # Check if the property is a data property and if its name is one of our desired parameters.
        # (You can alternatively check if the property name starts with "param" or similar.)
        # First, check if prop is an instance of a data property.
        try:
            from owlready2 import DataPropertyClass
            if isinstance(prop, DataPropertyClass) and prop.name in param_props:
                # Get all values (we assume one value per property in many cases)
                vals = list(prop[layer])
                # If only one value, store it directly, else as a list.
                if len(vals) == 1:
                    params[prop.name] = vals[0]
                elif vals:
                    params[prop.name] = vals
        except Exception:
            # If there is any error, skip this property.
            continue
    return params

def convert_network_to_json(network):
    """
    Given a network individual from ANNETT-O, traverse its layers (via 'hasLayer')
    and build a dictionary with the following structure:
      {
        "model": <network name>,
        "nodes": [
          {
            "op": <operation type derived from class name>,
            "name": <layer name>,
            "input": [list of input node names],
            "target": [list of output/target node names],
            "parameters": { parameter property: value, ... }
          },
          ...
        ]
      }
    """
    # Create the JSON structure
    net_dict = {"model": network.name, "nodes": []}
    
    # Get a list of layers.
    # (Assuming network has a property 'hasLayer'; adjust if needed.)
    layers = list(network.hasLayer) if hasattr(network, "hasLayer") else []
    
    # Optionally, sort layers if there is an ordering (e.g., by a sort key or by name).
    # For this example, we sort by layer.name
    layers.sort(key=lambda l: l.name)
    
    for layer in layers:
        # Derive the "operation" from the class name. You can include a mapping dictionary if needed.
        op = layer.__class__.__name__
        node = {
            "op": op,
            "name": layer.name,
            "input": [],
            "target": [],
            "parameters": {}
        }
        # Get input and target nodes from properties (if available)
        print(layer.previousLayer)
        node["input"] = [str(layer.name) for layer in layer.previousLayer ] #str(layer.previousLayer.iri)
        node["target"] = [str(layer.name) for layer in layer.nextLayer ]
        # Extract parameter data properties into a dictionary
        node["parameters"] = extract_layer_parameters(layer)
         
        net_dict["nodes"].append(node)
        
    return net_dict

# ----------------------- USAGE EXAMPLE -----------------------
if __name__ == "__main__":
    # Load your ontology (adjust path/URL accordingly)
    onto = get_ontology("/fastdata1/home/annett-o-0.1.owl").load()
    device = 'cuda'
    model = loadModel('fixed')
    # Suppose you have an ANNConfiguration that has networks.
    ann_config = onto.search_one(is_a=onto.ANNConfiguration).instances()
    
    if not ann_config:
        print("No ANNConfiguration instance found!")
    else:
        # For demonstration, let's assume we take the first network from this configuration.
        networks =   ann_config[0].__getattr__('hasNetwork') #list(ann_config.hasNetwork) if hasattr(ann_config, "hasNetwork") else []
        if not networks:
            print("No networks found under the ANNConfiguration!")
        else:
            for network in networks:
                print(network)
                net_json = [convert_network_to_json(network)]


                # Convert the dictionary to a JSON string and print it.
                json_str = json.dumps(net_json, indent=2)
                parsed_networks,string_parsed_networks,dataset = load_dataset('blah',json_str)
                print(get_embedding_dataset(model, 'fixed', dataset))
                print(json_str)


