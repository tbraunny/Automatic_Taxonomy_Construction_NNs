import logging
from owlready2 import Ontology, ThingClass, Thing, ObjectProperty, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties,
    get_connected_classes,
    get_subclasses,
    create_cls_instance, 
    assign_object_property_relationship, 
    create_subclass,
)
from utils.annetto_utils import int_to_ordinal, split_camel_case, get_right_of_last_underscore
from utils.llm_service import init_engine, query_llm
from fuzzywuzzy import fuzz
from json import load

# Classes to omit from instantiation.
OMIT_CLASSES = {"DataCharacterization", "Regularization"}


class OntologyInstantiator:
    """
    A class to instantiate an ontology by processing each component separately and linking them together.
    """
    def __init__(self, ontology: Ontology, json_file_path: str) -> None:
        self.ontology = ontology
        self.json_file_path = json_file_path
        self.llm_cache: dict[str, any] = {}  # Cache for LLM responses
        self.processed: set = set()
        self.full_context: list[Thing] = []
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _instantiate_cls(self, cls: ThingClass, instance_name: str) -> Thing:
        """
        Instantiate a given ontology class with the specified instance name.
        """
        instance_name = instance_name.replace(" ", "-").lower()
        unique_instance_name = f"{cls.name}_{instance_name}"
        instance = create_cls_instance(cls, unique_instance_name)
        self.logger.info(f"Instantiated {cls.name} with name: {unique_instance_name}")
        return instance

    def _link_instances(self, parent_instance: Thing, child_instance: Thing, object_property: ObjectProperty) -> None:
        """
        Link two instances via the provided object property.
        """
        assign_object_property_relationship(parent_instance, child_instance, object_property)
        self.logger.info(f"Linked {parent_instance.name} and {child_instance.name} via {object_property.name}.")

    def _query_llm(self, instructions: str, prompt: str) -> any:
        """
        Query the LLM with caching to avoid redundant calls.
        """
        full_prompt = f"{instructions}\n{prompt}"
        if full_prompt in self.llm_cache:
            self.logger.info(f"Using cached LLM response for prompt: {full_prompt}")
            print("Using cached LLM response #")
            return self.llm_cache[full_prompt]
        try:
            response = query_llm(full_prompt)
            self.logger.info(f"LLM query: {full_prompt}")
            self.logger.info(f"LLM query response: {response}")
            self.llm_cache[full_prompt] = response
            return response
        except Exception as e:
            self.logger.error(f"LLM query error: {e}")
            return ""
        
        
    
    def _find_ancestor_network_instance(self) -> Thing:
        """
        Return the network instance from the current context.
        """
        return next((thing for thing in self.full_context if self.ontology.Network in thing.is_a), None)

    def _process_objective_functions(self) -> None:
        """
        Process loss, cost, and regularizer functions, and link them to the network.
        """
        network_thing = self._find_ancestor_network_instance()
        if not network_thing:
            raise ValueError("No network instance found in the full context.")
        network_thing_name = network_thing.name

        loss_function_prompt = (
            f"Extract only the names of the loss functions used for the {network_thing_name}'s architecture and return the result in JSON format with the key 'answer'. "
            "Follow the examples below.\n\n"
            "Examples:\n"
            "Network: Discriminator\n"
            '{"answer": ["Binary Cross-Entropy Loss"]}\n'
            "Network: Discriminator\n"
            '{"answer": ["Wasserstein Loss", "Hinge Loss"]}\n\n'
            f"Now, for the following network:\nNetwork: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        loss_function_names = self._query_llm("", loss_function_prompt)
        if not loss_function_names:
            self.logger.info("No response for loss function classes.")
            return

        for loss_name in loss_function_names:
            loss_objective_prompt = (
                f"Is the {loss_name} function designed to minimize or maximize its objective function? "
                "Please respond with either 'minimize' or 'maximize' in JSON format using the key 'answer'.\n\n"
                "**Clarification:**\n"
                "- If the function is set up to minimize (e.g., cross-entropy, MSE), respond with 'minimize'.\n"
                "- If the function is set to maximize a likelihood or score function (e.g., log-likelihood, accuracy), respond with 'maximize'.\n"
                "- Note: Maximizing the log-probability typically corresponds to minimizing the negative log-likelihood.\n\n"
                "Examples:\n"
                "Loss Function: Cross-Entropy Loss\n"
                '{"answer": "minimize"}\n'
                "Loss Function: Custom Score Function\n"
                '{"answer": "maximize"}\n\n'
                f"Now, for the following loss function:\nLoss Function: {loss_name}\n"
                '{"answer": "<Your Answer Here>"}'
            )
            loss_obj_response = self._query_llm("", loss_objective_prompt)
            if not loss_obj_response:
                self.logger.info(f"No response for loss function objective for {loss_name}.")
                continue
            loss_obj_type = loss_obj_response.lower()

            if loss_obj_type == "minimize":
                objective_function_instance = self._instantiate_cls(
                    self.ontology.MinObjectiveFunction,
                    f"{network_thing_name}_min_objective_function"
                )
            elif loss_obj_type == "maximize":
                objective_function_instance = self._instantiate_cls(
                    self.ontology.MaxObjectiveFunction,
                    f"{network_thing_name}_max_objective_function"
                )
            else:
                self.logger.info(f"Invalid response for loss function objective for {loss_name}.")
                continue
            self.full_context.append(objective_function_instance)

            loss_function_instance = self._instantiate_cls(self.ontology.LossFunction, loss_name)
            cost_function_instance = self._instantiate_cls(
                self.ontology.CostFunction,
                f"{objective_function_instance.name}_cost_function"
            )

            self._link_instances(objective_function_instance, cost_function_instance, self.ontology.hasCost)
            self._link_instances(cost_function_instance, loss_function_instance, self.ontology.hasLoss)

            regularizer_function_prompt = (
                f"Extract only the names of explicit regularizer functions that are mathematically added to the objective function for the {loss_name} loss function. "
                "Exclude implicit regularization techniques like Dropout, Batch Normalization, or any regularization that is not directly part of the loss function. "
                "Return the result in JSON format with the key 'answer'. Follow the examples below.\n\n"
                "Examples:\n"
                "Loss Function: Discriminator Loss\n"
                '{"answer": ["L1 Regularization"]}\n\n'
                "Loss Function: Generator Loss\n"
                '{"answer": ["L2 Regularization", "Elastic Net"]}\n\n'
                "Loss Function: Cross-Entropy Loss\n"
                '{"answer": []}\n\n'
                "Loss Function: Binary Cross-Entropy Loss\n"
                '{"answer": ["L2 Regularization"]}\n\n'
                f"Now, for the following loss function:\nLoss Function: {loss_name}\n"
                '{"answer": "<Your Answer Here>"}'
            )
            regularizer_names = self._query_llm("", regularizer_function_prompt)
            if not regularizer_names:
                self.logger.info(f"No response for regularizer function classes for loss function {loss_name}.")
                continue
            if regularizer_names == []:
                self.logger.info(f"No regularizer functions provided for loss function {loss_name}.")
                continue

            for reg_name in regularizer_names:
                reg_instance = self._instantiate_cls(self.ontology.RegularizerFunction, reg_name)
                self._link_instances(cost_function_instance, reg_instance, self.ontology.hasRegularizer)

    def _process_layers(self) -> None:
        """
        Process the different layers (input, output, activation, noise, and modification) of the network.
        """
        network_thing = self._find_ancestor_network_instance()
        if not network_thing:
            raise ValueError("No network instance found in the full context.")
        network_thing_name = network_thing.name

        # Process Input Layer
        input_layer_prompt = (
            f"Extract the number of units in the input layer of the {network_thing_name} architecture. "
            "The number of units refers to the number of neurons or nodes in the input layer. "
            "Return the result as an integer in JSON format with the key 'answer'.\n\n"
            "Examples:\n"
            "1. Network: Discriminator\n"
            '{"answer": 784}\n\n'
            "2. Network: Generator\n"
            '{"answer": 100}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": 1}\n\n'
            f"Now, for the following network:\nNetwork: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        input_units = self._query_llm("", input_layer_prompt)
        if not input_units:
            self.logger.info("No response for input layer units.")
        else:
            input_layer_instance = self._instantiate_cls(self.ontology.InputLayer, f"{network_thing_name} Input Layer")
            input_layer_instance.layer_num_units = [input_units]
            self._link_instances(network_thing, input_layer_instance, self.ontology.hasLayer)

        # Process Output Layer
        output_layer_prompt = (
            f"Extract the number of units in the output layer of the {network_thing_name} architecture. "
            "The number of units refers to the number of neurons or nodes in the output layer. "
            "Return the result as an integer in JSON format with the key 'answer'.\n\n"
            "Examples:\n"
            "1. Network: Discriminator\n"
            '{"answer": 1}\n\n'
            "2. Network: Generator\n"
            '{"answer": 784}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": 1}\n\n'
            f"Now, for the following network:\nNetwork: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        output_units = self._query_llm("", output_layer_prompt)
        if not output_units:
            self.logger.info("No response for output layer units.")
        else:
            output_layer_instance = self._instantiate_cls(self.ontology.OutputLayer, f"{network_thing_name} Output Layer")
            output_layer_instance.layer_num_units = [output_units]
            self._link_instances(network_thing, output_layer_instance, self.ontology.hasLayer)

        # Process Activation Layers
        activation_layer_prompt = (
            f"Extract the number of instances of each core layer type in the {network_thing_name} architecture. "
            "Only count layers that represent essential network operations such as convolutional layers, "
            "fully connected (dense) layers, and attention layers.\n"
            "Do NOT count layers that serve as noise layers (i.e. guassian, normal, etc), "
            "activation functions (e.g., ReLU, Sigmoid), or modification layers (e.g., dropout, batch normalization), "
            "or pooling layers (e.g. max pool, average pool).\n\n"
            "Please provide the output in JSON format using the key \"answer\", where the value is a dictionary "
            "mapping the layer type names to their counts.\n\n"
            "Examples:\n\n"
            "1. Network Architecture Description:\n"
            "- 3 Convolutional layers\n"
            "- 2 Fully Connected layers\n"
            "- 2 Recurrent layers\n"
            "- 1 Attention layer\n"
            "- 3 Transformer Encoder layers\n"
            "Expected JSON Output:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"Convolutional\": 3,\n"
            "    \"Fully Connected\": 2,\n"
            "    \"Recurrent\": 2,\n"
            "    \"Attention\": 1,\n"
            "    \"Transformer Encoder\": 3\n"
            "  }\n"
            "}\n\n"
            "2. Network Architecture Description:\n"
            "- 3 Convolutional layers\n"
            "- 2 Fully Connected layer\n"
            "- 2 Recurrent layer\n"
            "- 1 Attention layers\n"
            "- 3 Transformer Encoder layers\n"
            "Expected JSON Output:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"Convolutional\": 4,\n"
            "    \"FullyConnected\": 1,\n"
            "    \"Recurrent\": 2,\n"
            "    \"Attention\": 1,\n"
            "    \"Transformer Encoder\": 3\n"
            "  }\n"
            "}\n\n"
            "Now, for the following network:\n"
            f"Network: {network_thing_name}\n"
            "Expected JSON Output:\n"
            "{\n"
            "  \"answer\": \"<Your Answer Here>\"\n"
            "}\n"
        )
        activation_layer_counts = self._query_llm("", activation_layer_prompt)
        if not activation_layer_counts:
            self.logger.info("No response for activation layer classes.")
        else:
            for layer_type, layer_count in activation_layer_counts.items():
                for i in range(layer_count):
                    layer_instance = self._instantiate_cls(self.ontology.ActivationLayer, f"{layer_type} {i + 1}")
                    self._link_instances(network_thing, layer_instance, self.ontology.hasLayer)
                    # Process bias for activation layer
                    layer_ordinal = int_to_ordinal(i + 1)
                    bias_prompt = (
                        f"Does the {layer_ordinal} {layer_type} layer include a bias term? "
                        "Please respond with either 'true', 'false', or an empty list [] if unknown, in JSON format using the key 'answer'.\n\n"
                        "Clarification:\n"
                        "- A layer has a bias term if it adds a constant (bias) to the weighted sum before applying the activation function.\n"
                        "- Examples of layers that often include bias: Fully Connected (Dense) layers, Convolutional layers.\n"
                        "- Some layers like Batch Normalization typically omit bias.\n\n"
                        "Examples:\n"
                        "1. Layer: Fully-Connected\n"
                        '{"answer": "true"}\n\n'
                        "2. Layer: Convolutional\n"
                        '{"answer": "true"}\n\n'
                        "3. Layer: Attention\n"
                        '{"answer": "false"}\n\n'
                        "4. Layer: UnknownLayerType\n"
                        '{"answer": []}\n\n'
                        f"Now, for the following layer:\nLayer: {layer_ordinal} {layer_type}\n"
                        '{"answer": "<Your Answer Here>"}'
                    )
                    has_bias_response = self._query_llm("", bias_prompt)
                    if has_bias_response:
                        if has_bias_response.lower() == "true":
                            layer_instance.has_bias = [True]
                        elif has_bias_response.lower() == "false":
                            layer_instance.has_bias = [False]
                        self.logger.info(f"Set bias term for {layer_ordinal} {layer_type} to {layer_instance.has_bias}.")

                    # Process activation function for activation layer
                    activation_function_prompt = (
                        f"Goal:\nIdentify the activation function used in the {layer_ordinal} {layer_type} layer, if any.\n\n"
                        "Return Format:\nRespond with the activation function name in JSON format using the key 'answer'. If there is no activation function or it's unknown, return an empty list [].\n"
                        "Examples:\n"
                        '{"answer": "ReLU"}\n'
                        '{"answer": "Sigmoid"}\n'
                        '{"answer": []}\n\n'
                        f"Now, for the following layer:\nLayer: {layer_ordinal} {layer_type}\n"
                        '{"answer": "<Your Answer Here>"}'
                    )
                    activation_function_response = self._query_llm("", activation_function_prompt)
                    if activation_function_response:
                        if activation_function_response != "[]":
                            activation_function_instance = self._instantiate_cls(
                                self.ontology.ActivationFunction,
                                activation_function_response
                            )
                            self._link_instances(layer_instance, activation_function_instance, self.ontology.hasActivationFunction)
                        else:
                            self.logger.info(f"No activation function associated with {layer_ordinal} {layer_type}.")

        # Process Noise Layers
        noise_layer_prompt = (
            f"Does the {network_thing_name} architecture include any noise layers? "
            "Noise layers are layers that introduce randomness or noise into the network. "
            "Examples include Dropout, Gaussian Noise, and Batch Normalization. "
            "Please respond with either 'true' or 'false' in JSON format using the key 'answer'.\n\n"
            "Examples:\n"
            "1. Network: Discriminator\n"
            '{"answer": "true"}\n\n'
            "2. Network: Generator\n"
            '{"answer": "false"}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": "true"}\n\n'
            f"Now, for the following network:\nNetwork: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        noise_layer_response = self._query_llm("", noise_layer_prompt)
        if not noise_layer_response:
            self.logger.info("No response for noise layer classes.")
        elif noise_layer_response.lower() == "true":
            noise_layer_pdf_prompt = (
                f"Extract the probability distribution function (PDF) and its associated hyperparameters for the noise layers in the {network_thing_name} architecture. "
                "Noise layers introduce randomness or noise into the network. "
                "Examples include Dropout, Gaussian Noise, and Batch Normalization. "
                "Return the result in JSON format with the key 'answer'.\n\n"
                "Examples:\n"
                "1. Network: Discriminator\n"
                '{"answer": {"Dropout": {"rate": 0.5}}}\n\n'
                "2. Network: Generator\n"
                '{"answer": {"Gaussian Noise": {"mean": 0, "stddev": 1}}}\n\n'
                "3. Network: Linear Regression\n"
                '{"answer": {"Dropout": {"rate": 0.3}}}\n\n'
                f"Now, for the following network:\nNetwork: {network_thing_name}\n"
                '{"answer": "<Your Answer Here>"}'
            )
            noise_layer_pdf = self._query_llm("", noise_layer_pdf_prompt)
            if not noise_layer_pdf:
                self.logger.info("No response for noise layer PDF.")
            else:
                for noise_name, noise_params in noise_layer_pdf.items():
                    noise_layer_instance = self._instantiate_cls(self.ontology.NoiseLayer, noise_name)
                    self._link_instances(network_thing, noise_layer_instance, self.ontology.hasLayer)
                    for param_name, param_value in noise_params.items():
                        setattr(noise_layer_instance, param_name, [param_value])
            # Process Modification Layers
            modification_layer_prompt = (
                f"Extract the number of instances of each modification layer type in the {network_thing_name} architecture. "
                "Modification layers include layers that alter the input data or introduce noise, such as Dropout, Batch Normalization, and Layer Normalization. "
                "Exclude noise layers (e.g., Gaussian Noise, Dropout) and activation layers (e.g., ReLU, Sigmoid) from your count.\n"
                "Please provide the output in JSON format using the key \"answer\", where the value is a dictionary "
                "mapping the layer type names to their counts.\n\n"
                "Examples:\n\n"
                "1. Network Architecture Description:\n"
                "- 3 Dropout layers\n"
                "- 2 Batch Normalization layers\n"
                "- 1 Layer Normalization layer\n"
                "Expected JSON Output:\n"
                "{\n"
                "  \"answer\": {\n"
                "    \"Dropout\": 3,\n"
                "    \"Batch Normalization\": 2,\n"
                "    \"Layer Normalization\": 1\n"
                "  }\n"
                "}\n\n"
                "2. Network Architecture Description:\n"
                "- 3 Dropout layers\n"
                "- 2 Batch Normalization layers\n"
                "- 1 Layer Normalization layer\n"
                "Expected JSON Output:\n"
                "{\n"
                "  \"answer\": {\n"
                "    \"Dropout\": 3,\n"
                "    \"Batch Normalization\": 2,\n"
                "    \"Layer Normalization\": 1\n"
                "  }\n"
                "}\n\n"
                "Now, for the following network:\n"
                f"Network: {network_thing_name}\n"
                "Expected JSON Output:\n"
                "{\n"
                "  \"answer\": \"<Your Answer Here>\"\n"
                "}\n"
            )
            modification_layer_counts = self._query_llm("", modification_layer_prompt)
            if not modification_layer_counts:
                self.logger.info("No response for modification layer classes.")
            else:
                dropout_match = next(
                    (s for s in modification_layer_counts if fuzz.token_set_ratio("dropout", s) >= 85),
                    None
                )
                dropout_layer_rate = None
                if dropout_match:
                    dropout_rate_prompt = (
                        f"Extract the dropout rate for the Dropout layers in the {network_thing_name} architecture. "
                        "The dropout rate is the fraction of input units to drop during training. "
                        "Return the result as a float in JSON format with the key 'answer'.\n\n"
                        "Examples:\n"
                        "1. Network: Discriminator\n"
                        '{"answer": 0.5}\n\n'
                        "2. Network: Generator\n"
                        '{"answer": 0.3}\n\n'
                        "3. Network: Linear Regression\n"
                        '{"answer": 0.2}\n\n'
                        f"Now, for the following network:\nNetwork: {network_thing_name}\n"
                        '{"answer": "<Your Answer Here>"}'
                    )
                    dropout_layer_rate = self._query_llm("", dropout_rate_prompt)
                    if not dropout_layer_rate:
                        self.logger.info("No response for dropout layer rate.")
                for layer_type, layer_count in modification_layer_counts.items():
                    for i in range(layer_count):
                        if dropout_match and layer_type == dropout_match:
                            layer_instance = self._instantiate_cls(self.ontology.DropoutLayer, f"{layer_type} {i + 1}")
                            if dropout_layer_rate:
                                layer_instance.dropout_rate = [dropout_layer_rate]
                            self._link_instances(network_thing, layer_instance, self.ontology.hasLayer)
                        else:
                            layer_instance = self._instantiate_cls(self.ontology.ModificationLayer, f"{layer_type} {i + 1}")
                            self._link_instances(network_thing, layer_instance, self.ontology.hasLayer)

    def _process_task_characterization(self) -> None:
        """
        Process the task characterization of the network.
        """
        network_thing = self._find_ancestor_network_instance()
        if not network_thing:
            raise ValueError("No network instance found in the full context.")
        network_thing_name = network_thing.name

        task_prompt = (
            "Extract the primary task that the given network architecture is designed to perform. "
            "The primary task is the most important or central objective of the network. "
            "Return the task name in JSON format with the key 'answer'.\n\n"
            "The types of tasks include:\n"
            "- Adversarial\n"
            "- Classification\n"
            "- SemiSupervised Classification\n"
            "- Supervised Classification\n"
            "- Clustering\n"
            "- Discrimination\n"
            "- Generation\n"
            "- Reconstruction\n"
            "- Regression\n\n"
            "Examples:\n"
            "1. Discrimination\n"
            '{"answer": "Discriminator"}\n\n'
            "2. Network: Generator\n"
            '{"answer": "Generation"}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": "Regression"}\n\n'
            f"Now, for the following network:\nNetwork: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        task_name = self._query_llm("", task_prompt)
        if not task_name:
            self.logger.info("No response for task characterization.")
        else:
            task_instance = self._instantiate_cls(self.ontology.TaskCharacterization, task_name)
            self._link_instances(network_thing, task_instance, self.ontology.hasTaskType)

    def _process_entity(self, cls: ThingClass) -> None:
        """
        Process a given ontology class entity if it has not already been processed.
        """
        if cls in self.processed or cls.name in OMIT_CLASSES:
            return
        self.processed.add(cls)
        if cls == self.ontology.Layer:
            self._process_layers()
        elif cls == self.ontology.ObjectiveFunction:
            self._process_objective_functions()
        elif cls == self.ontology.TaskCharacterization:
            self._process_task_characterization()

    def run(self) -> None:
        """
        Main method to run the ontology instantiation process.
        """
        # Verify required root exists.
        if not hasattr(self.ontology, 'ANNConfiguration'):
            self.logger.error("Error: Class 'ANNConfiguration' not found in ontology.")
            return

        # Initialize the LLM engine with the document context.
        init_engine(self.json_file_path)

        # Extract the title from the JSON file.
        title = "AlexNet"
        # try:
        #     with open(self.json_file_path, 'r', encoding='utf-8') as file:
        #         data = load(file)
        #     titles = [item['metadata']['title'] for item in data if 'metadata' in item and 'title' in item['metadata']]
        #     title = titles[0] if titles else "DefaultTitle"
        # except Exception as e:
        #     self.logger.error(f"Error reading JSON file: {e}")
        #     title = "DefaultTitle"

        self.processed.add(self.ontology.TrainingStrategy)  # Temporary fix to avoid infinite loop
        self.processed.add(self.ontology.ANNConfiguration)
        ann_config_instance = self._instantiate_cls(self.ontology.ANNConfiguration, title)
        self.full_context.append(ann_config_instance)

        # Process the top-level Network class.
        if hasattr(self.ontology, "Network"):
            network_instance = self._instantiate_cls(self.ontology.Network, "Convolutional Network")
            self._link_instances(ann_config_instance, network_instance, self.ontology.hasNetwork)
            self.full_context.append(network_instance)
            for connected_class in get_connected_classes(self.ontology.Network, self.ontology):
                if isinstance(connected_class, ThingClass):
                    self._process_entity(connected_class)

        self.logger.info("An ANN has been successfully instantiated.")


if __name__ == "__main__":
    OUTPUT_FILE = './test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()
    with ontology:
        instantiator = OntologyInstantiator(ontology, "data/alexnet/doc_alexnet.json")
        instantiator.run()
        new_file_path = "data/owl/annett-o-test.owl"
        ontology.save(file=new_file_path, format="rdfxml")
        logging.info(f"Ontology saved to {new_file_path}")

