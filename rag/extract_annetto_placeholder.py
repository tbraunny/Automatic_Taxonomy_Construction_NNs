import json
from llama_index.core import VectorStoreIndex

# This is all ChatGPT garabage placeholder for now

# Function to query and return atomic answers
def query_and_parse_atomic(query_engine, query):

    query += "You are a system designed to extract detailed and atomic information about convolutional neural network (CNN) architectures.\
Your task is to answer in concise and structured formats. Do not elaborate with extra context. If you are unsure, respond with 'I don't know'.\
Focus on providing specific values or descriptions for each element.\
For example:\
- If asked for the 'Number of Layers', respond with a simple number like '8'.\
- If asked for 'Filter Sizes', list them in the format: '11x11, 5x5, 3x3'.\
- If asked a yes or no type question, answer 'Yes' or 'No'\
- If a layer type or parameter is not explicitly mentioned, respond with 'I don't know'." + query
    response = query_engine.query(query)
    response_text = response.text if hasattr(response, "text") else str(response)
    
    # Check for an empty or uninformative response and return "Not Available"
    if not response_text.strip() or "I don't know" in response_text:
        return "N/A"

    # Use simple parsing logic to strip sentences and extract key values
    # This can be improved with regular expressions if necessary
    return response_text.strip()

# Parsing functions for atomic details

def extract_topology_info(query_engine):
    topology_info = {
        "NumberOfLayers": query_and_parse_atomic(query_engine, "How many layers does the network have?"),
        "LayerTypes": query_and_parse_atomic(query_engine, "List the layer types (e.g., convolutional, pooling, fully connected)."),
        "FilterSizes": query_and_parse_atomic(query_engine, "List the filter sizes for the convolutional layers."),
        "Stride": query_and_parse_atomic(query_engine, "What is the stride for the convolutional layers?"),
        "Padding": query_and_parse_atomic(query_engine, "What type of padding is used in the convolutional layers?"),
        "PoolingDetails": query_and_parse_atomic(query_engine, "Describe the pooling operations, including sizes and strides.")
    }
    print("Topology Information:\n", topology_info)
    return topology_info

def extract_activation_functions(query_engine):
    activation_info = {
        "ActivationFunction": query_and_parse_atomic(query_engine, "What activation function is used?"),
        "ActivationFunctionPerLayer": query_and_parse_atomic(query_engine, "Specify the activation function for each layer.")
    }
    print("Activation Functions:\n", activation_info)
    return activation_info

def extract_layer_connectivity(query_engine):
    connectivity_info = {
        "ConnectionSequence": query_and_parse_atomic(query_engine, "Provide the connection sequence of layers."),
        "SkipConnections": query_and_parse_atomic(query_engine, "Are there any skip connections? If so, describe them."),
        "LayerDependencies": query_and_parse_atomic(query_engine, "Explain any dependencies between layers.")
    }
    print("Layer Connectivity:\n", connectivity_info)
    return connectivity_info

def extract_training_details(query_engine):
    training_info = {
        "Algorithm": query_and_parse_atomic(query_engine, "Which optimization algorithm is used?"),
        "LearningRate": query_and_parse_atomic(query_engine, "What is the learning rate?"),
        "Momentum": query_and_parse_atomic(query_engine, "What is the momentum value, if used?"),
        "WeightDecay": query_and_parse_atomic(query_engine, "What is the weight decay value?"),
        "BatchSize": query_and_parse_atomic(query_engine, "What batch size is used?"),
        "Epochs": query_and_parse_atomic(query_engine, "How many epochs are used for training?"),
        "Dropout": query_and_parse_atomic(query_engine, "Is dropout used, and what is the rate?")
    }
    print("Training Details:\n", training_info)
    return training_info

def extract_training_dataset(query_engine):
    dataset_info = {
        "Name": query_and_parse_atomic(query_engine, "What is the name of the dataset?"),
        "Size": query_and_parse_atomic(query_engine, "How many samples are in the dataset?"),
        "Classes": query_and_parse_atomic(query_engine, "How many classes are in the dataset?")
    }
    print("Training Dataset:\n", dataset_info)
    return dataset_info

def extract_evaluation_metrics(query_engine):
    evaluation_info = {
        "Top1ErrorRate": query_and_parse_atomic(query_engine, "What is the top-1 error rate?"),
        "Top5ErrorRate": query_and_parse_atomic(query_engine, "What is the top-5 error rate?"),
        "OtherMetrics": query_and_parse_atomic(query_engine, "List any other evaluation metrics reported.")
    }
    print("Evaluation Metrics:\n", evaluation_info)
    return evaluation_info

# Function to collect and parse all information
def collect_and_parse_info(index, llm_predictor):
    query_engine = index.as_query_engine(llm=llm_predictor)

    # Collect all atomic details
    topology_info = extract_topology_info(query_engine)
    activation_functions = extract_activation_functions(query_engine)
    layer_connectivity = extract_layer_connectivity(query_engine)
    training_details = extract_training_details(query_engine)
    training_dataset = extract_training_dataset(query_engine)
    evaluation_metrics = extract_evaluation_metrics(query_engine)

    # Structure data in a format aligned with ANNETT-O ontology
    annetto_data = {
        "@context": {
            "@vocab": "http://w3id.org/annett-o#"
        },
        "@graph": [
            {
                "@type": "ANNConfiguration",
                "hasTopology": topology_info,
                "hasActivationFunction": activation_functions,
                "hasLayerConnectivity": layer_connectivity,
                "hasTraining": {
                    "@type": "Training",
                    "Details": training_details,
                    "usesDataset": training_dataset
                },
                "hasEvaluation": evaluation_metrics
            }
        ]
    }

    # Convert to JSON for OWL format compliance
    owl_json = json.dumps(annetto_data, indent=4)
    # print("Generated OWL-compliant JSON:\n", owl_json)
    return owl_json

# Main function
def prompt_engr(index: VectorStoreIndex, llm_predictor):
    print("Extracting atomic information for CNN paper...")
    owl_json = collect_and_parse_info(index, llm_predictor)
    # Save JSON to a file
    with open("network_info.json", "w") as file:
        file.write(owl_json)
    print("Data successfully written to network_info.json")
