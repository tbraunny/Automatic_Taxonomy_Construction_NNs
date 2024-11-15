import json
from llama_index.core import VectorStoreIndex

'''
Example Usage:

from utils.cnn_info_extractor import PromptEngr

extractor = PromptEngr(index, llm_predictor)
extractor.save_info_to_file("network_info.json")

'''
class PromptEngr:
    """
    A utility class to extract detailed and structured information about CNN architectures.
    """

    def __init__(self, index: VectorStoreIndex, llm_predictor):
        """
        Initialize the CNNInfoExtractor with a vector store index and an LLM predictor.
        :param index: A vector store index used for querying.
        :param llm_predictor: The LLM predictor used for generating responses.
        """
        self.query_engine = index.as_query_engine(llm=llm_predictor)

    def _query_and_parse_atomic(self, query):
        """
        Query and parse atomic answers from the LLM query engine.
        :param query: The query string.
        :return: A parsed response string or 'N/A' if no valid response is received.
        """
        prompt = (
            query +
            " You are a system designed to extract detailed and atomic information about convolutional neural network (CNN) architectures. "
            "Your task is to answer in concise and structured formats. Do not elaborate with extra context. If you are unsure, respond with 'I don't know'. "
            "Focus on providing specific values or descriptions for each element. "
            "For example: "
            "- If asked for the 'Number of Layers', respond with a simple number like '8'. "
            "- If asked for 'Filter Sizes', list them in the format: '11x11, 5x5, 3x3'. "
            "- If asked a yes or no type question, answer 'Yes' or 'No'. "
            "- If a layer type or parameter is not explicitly mentioned, respond with 'I don't know'."
        )
        response = self.query_engine.query(prompt)
        response_text = response.text if hasattr(response, "text") else str(response)

        if not response_text.strip() or "I don't know" in response_text:
            return "N/A"

        return response_text.strip()

    def extract_topology_info(self):
        return {
            "NumberOfLayers": self._query_and_parse_atomic("How many layers does the network have?"),
            "LayerTypes": self._query_and_parse_atomic("List the layer types (e.g., convolutional, pooling, fully connected)."),
            "FilterSizes": self._query_and_parse_atomic("List the filter sizes for the convolutional layers."),
            "Stride": self._query_and_parse_atomic("What is the stride for the convolutional layers?"),
            "Padding": self._query_and_parse_atomic("What type of padding is used in the convolutional layers?"),
            "PoolingDetails": self._query_and_parse_atomic("Describe the pooling operations, including sizes and strides.")
        }

    def extract_activation_functions(self):
        return {
            "ActivationFunction": self._query_and_parse_atomic("What activation function is used?"),
            "ActivationFunctionPerLayer": self._query_and_parse_atomic("Specify the activation function for each layer.")
        }

    def extract_layer_connectivity(self):
        return {
            "ConnectionSequence": self._query_and_parse_atomic("Provide the connection sequence of layers."),
            "SkipConnections": self._query_and_parse_atomic("Are there any skip connections? If so, describe them."),
            "LayerDependencies": self._query_and_parse_atomic("Explain any dependencies between layers.")
        }

    def extract_training_details(self):
        return {
            "Algorithm": self._query_and_parse_atomic("Which optimization algorithm is used?"),
            "LearningRate": self._query_and_parse_atomic("What is the learning rate?"),
            "Momentum": self._query_and_parse_atomic("What is the momentum value, if used?"),
            "WeightDecay": self._query_and_parse_atomic("What is the weight decay value?"),
            "BatchSize": self._query_and_parse_atomic("What batch size is used?"),
            "Epochs": self._query_and_parse_atomic("How many epochs are used for training?"),
            "Dropout": self._query_and_parse_atomic("Is dropout used, and what is the rate?")
        }

    def extract_training_dataset(self):
        return {
            "Name": self._query_and_parse_atomic("What is the name of the dataset?"),
            "Size": self._query_and_parse_atomic("How many samples are in the dataset?"),
            "Classes": self._query_and_parse_atomic("How many classes are in the dataset?")
        }

    def extract_evaluation_metrics(self):
        return {
            "Top1ErrorRate": self._query_and_parse_atomic("What is the top-1 error rate?"),
            "Top5ErrorRate": self._query_and_parse_atomic("What is the top-5 error rate?"),
            "OtherMetrics": self._query_and_parse_atomic("List any other evaluation metrics reported.")
        }

    def collect_and_parse_info(self):
        """
        Collect all information and structure it into an OWL-compliant JSON format.
        :return: A JSON string representing the structured information.
        """
        topology_info = self.extract_topology_info()
        activation_functions = self.extract_activation_functions()
        layer_connectivity = self.extract_layer_connectivity()
        training_details = self.extract_training_details()
        training_dataset = self.extract_training_dataset()
        evaluation_metrics = self.extract_evaluation_metrics()

        annetto_data = {
            "@context": {"@vocab": "http://w3id.org/annett-o#"},
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

        return json.dumps(annetto_data, indent=4)

    def save_info_to_file(self, file_path="network_info.json"):
        """
        Save the collected information to a JSON file.
        :param file_path: The file path where the JSON should be saved.
        """
        owl_json = self.collect_and_parse_info()
        with open(file_path, "w") as file:
            file.write(owl_json)
        print(f"Data successfully written to {file_path}")
