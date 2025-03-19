import logging
import os
import json
import hashlib
from datetime import datetime
import time
from typing import Dict, Any, Union, List, Optional
from utils.onnx_db import OnnxAddition

from owlready2 import Ontology, ThingClass, Thing, ObjectProperty, get_ontology
from rapidfuzz import process , fuzz
from pydantic import BaseModel
import warnings
from utils.constants import Constants as C
from utils.owl_utils import (
    create_cls_instance,
    assign_object_property_relationship,
    create_subclass,
    get_all_subclasses,
)
from utils.annetto_utils import int_to_ordinal, make_thing_classes_readable

# from utils.llm_service import init_engine, query_llm
from utils.llm_service_josue import init_engine, query_llm
from utils.pydantic_models import *

# Set up logging
log_dir = "logs"
log_file = os.path.join(
    log_dir, f"instantiate_annetto_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
)
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write to file
        # logging.StreamHandler()  # Print to console
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class OntologyInstantiator:
    """
    A class to instantiate an annett-o ontology by processing each main component separately and linking them together.
    """

    def __init__(
        self,
        ontology_path: str,
        list_json_doc_paths: List[str],
        ann_config_name: str = "alexnet",
        output_owl_path: str = "data/annett-o-test.owl",
    ) -> None:
        """
        Initialize the OntologyInstantiator class.
        # Args:
            ontology_path (str): The path to the ontology file.
            json_doc_files_paths (list[str]): The list of str paths to the JSON_doc files for paper and/or code.
            ann_config_name (str): The name of the ANN configuration.
            output_owl_path (str): The path to save the output OWL file.
        """
        if not isinstance(ann_config_name, str):
            self.logger.error("Expected a string for ANN Configuration name.")
            raise TypeError("Expected a string for ANN Configuration name.")
        if not isinstance(ontology_path, str):
            self.logger.error("Expected a string for ontology path.")
            raise TypeError("Expected a string for ontology path.")
        if not isinstance(list_json_doc_paths, list) and all(
            isinstance(path, str) for path in list_json_doc_paths
        ):
            self.logger.error("Expected a list of strings for JSON doc paths.")
            raise TypeError("Expected a list of strings for JSON doc paths.")
        if not isinstance(output_owl_path, str):
            self.logger.error("Expected a string for output OWL path.")
            raise TypeError("Expected a string for output OWL path.")
        
        if not ontology_path:
            self.ontology = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
        else:
            self.ontology = get_ontology(ontology_path).load()
        self.list_json_doc_paths = list_json_doc_paths
        self.llm_cache: Dict[str, Any] = {}
        self.logger = logger
        self.ann_config_name = ann_config_name.lower().strip()
        self.ann_config_hash = self._generate_hash(self.ann_config_name)
        self.output_owl_path = output_owl_path

    def _generate_hash(self, str: str) -> str:
        """
        Generate a unique hash identifier based on the given string.
        """
        hash_object = hashlib.md5(str.encode())  # Generate a consistent hash
        return hash_object.hexdigest()[:8]

    def _instantiate_and_format_class(
        self, cls: ThingClass, instance_name: str
    ) -> Thing:
        """
        Instantiate a given ontology class with the specified instance name.
        Uses the ANN configuration hash as a prefix for uniqueness.
        """
        unique_instance_name = self._hash_and_format_instance_name(instance_name)
        instance = create_cls_instance(cls, unique_instance_name)
        self.logger.info(
            f"Instantiated {cls.name} with name: {self._unhash_and_format_instance_name(unique_instance_name)}."
        )
        return instance

    def _hash_and_format_instance_name(self, instance_name: str) -> str:
        """
        Generate a unique instance name using the hash of the ANN config name.

        Ensures instance names remain consistent for a given ANN config name
        while maintaining readability.

        Example:
            Input: "Convolutional Layer"
            Output: "abcd1234_convolutional-layer" (assuming the hash is abcd1234)

        Args:
            instance_name (str): The base name of the instance.

        Returns:
            str: A unique instance name prefixed with the ANN config hash.
        """
        return f"{self.ann_config_hash}_{instance_name.replace(' ', '-').lower()}"

    def _unhash_and_format_instance_name(self, instance_name: str) -> str:
        """
        Remove the ANN config hash prefix from the instance name and restore readability.

        This method extracts the actual instance name by stripping out the
        prefixed hash and replacing dashes with spaces.

        Example: Input: "abcd1234_convolutional-layer" Output: "Convolutional Layer"

        Args:
            instance_name (str): The unique instance name with the hash prefix.

        Returns:
            str: The original readable instance name.
        """
        parts = instance_name.split("_", 1)  # Split at the first underscore
        stripped_name = parts[-1]  # Extract the actual instance name (without hash)
        return stripped_name.replace("-", " ")  # Convert dashes back to spaces

    ###### NOTE
    """
    Changed fuzzy match to be case insensitive 
    (was returning scores of 50 for identical strings w differing capitalization)
    """
    def _fuzzy_match_class(
        self, instance_name: str, classes: List[ThingClass], threshold: int = 80
    ) -> Optional[ThingClass]:
        """
        Perform fuzzy matching to find the best match for an instance to a list ThingClass's.

        :param instance_name: The instance name.
        :param classes: A list of ThingClass objects to match with.
        :param threshold: The minimum score required for a match.
        :return: The best-matching ThingClass object or None if no good match is found.
        """
        if not isinstance(instance_name, str):
            raise TypeError("Expected instance_name to be a string.")
        if not all(isinstance(cls, ThingClass) for cls in classes):
            raise TypeError("Expected classes to be a list of ThingClass objects.")
        if not all(isinstance(cls.name, str) for cls in classes):
            raise TypeError("Expected classes to have string names. ######")
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.")

        # Convert classes to a dictionary for lookup
        class_name_map = {cls.name.lower(): cls for cls in classes}

        match, score, _ = process.extractOne(
            instance_name.lower(), class_name_map.keys(), scorer=fuzz.ratio
        )

        return class_name_map[match] if score >= threshold else None

    def _fuzzy_match_list(self , class_names: List[str] , instance=None , threshold: int = 80) -> Optional[str]:
        """
        Perform fuzzy matching to find the best match for an instance in a list of strings.

        :param class_names: A list of string names to match with.
        :param instance_name: The instance name.
        :param threshold: The minimum score required for a match.
        :return: The best-matching string or None if no good match is found.
        """
        if not instance:
            instance = self.ann_config_name

        if not all(isinstance(name, str) for name in class_names):
            raise TypeError("Expected class_names to be a list of strings.")
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.")
        
        class_names_lower = [name.lower() for name in class_names]
        match, score, _ = process.extractOne(self.ann_config_name.lower(), class_names_lower, scorer=fuzz.ratio)

        return match if score >= threshold else None

    def _link_instances(
        self,
        parent_instance: Thing,
        child_instance: Thing,
        object_property: ObjectProperty,
    ) -> None:
        """
        Link two instances via an object property.
        """
        assign_object_property_relationship(
            parent_instance, child_instance, object_property
        )
        self.logger.info(
            f"Linked {self._unhash_and_format_instance_name(parent_instance.name)} and {self._unhash_and_format_instance_name(child_instance.name)} via {object_property.name}."
        )

    def _query_llm(
        self,
        instructions: str,
        prompt: str,
        json_format_instructions: Optional[str],
        pydantic_type_schema: Optional[type[BaseModel]],
    ) -> Union[Dict[str, Any], int, str, List[str]]:
        """
        Queries the LLM with a structured prompt to obtain a response in a specific format.

        The prompt should include few-shot examples demonstrating the expected structure of the output.
        The LLM is expected to return a JSON object where the primary key is "answer", and the value
        can be one of the following types:
        - Integer (e.g., {"answer": 100})
        - String (e.g., {"answer": "ReLU Activation"})
        - List of strings (e.g., {"answer": ["L1 Regularization", "Dropout"]})
        - Dictionary mapping strings to integers (e.g., {"answer": {"Convolutional": 4, "FullyConnected": 1}})

        If both `json_format_instructions` and `pydantic_type_schema` are provided, the function will
        parse the LLM's response and return it as an instance of the provided Pydantic class, ensuring that
        the output conforms to the expected schema.

        The function checks for cached responses before querying the LLM.
        If an error occurs, it logs the error and returns an empty response.

        Example prompt structure:

        Examples:
        Loss Function: Discriminator Loss
        1. Network: Discriminator
        {"answer": 784}

        2. Network: Generator
        {"answer": 100}

        3. Network: Linear Regression
        {"answer": 1}

        Now, for the following network:
        Network: {network_thing_name}
        Expected JSON Output:
        {"answer": "<Your Answer Here>"}

        Loss Function: Generator Loss
        {"answer": ["L2 Regularization", "Elastic Net"]}

        Loss Function: Cross-Entropy Loss
        {"answer": []}

        Loss Function: Binary Cross-Entropy Loss
        {"answer": ["L2 Regularization"]}

        Now, for the following loss function:
        Loss Function: {loss_name}
        {"answer": "<Your Answer Here>"}

        Args:
            instructions (str): Additional guidance for formatting the response.
            prompt (str): The main query containing the few-shot examples.
            json_format_instructions (Optional[str]): Additional JSON formatting instructions.
            pydantic_type_schema (Optional[type[BaseModel]]): A Pydantic model class that defines the expected output schema.

        Returns:
            Union[dict, int, str, list[str]]: The parsed LLM response based on the provided examples.
            If both `json_format_instructions` and `pydantic_type_schema` are provided, the response will be
            returned as an instance of the provided Pydantic class.
        """
        full_prompt = f"{instructions}\n{prompt}"
        if full_prompt in self.llm_cache:
            self.logger.info(f"Using cached LLM response for prompt: {full_prompt}")
            print("Using cached LLM response #####")
            return self.llm_cache[full_prompt]
        try:
            # Response returned as pydantic class if json_format_instructions and pydantic_type_schema are provided.
            response = query_llm(
                self.ann_config_name,
                full_prompt,
                json_format_instructions,
                pydantic_type_schema,
            )

            self.logger.info(f"LLM query: {full_prompt}")
            self.logger.info(f"LLM query response: {response}")
            self.llm_cache[full_prompt] = response

            return response
        except Exception as e:
            self.logger.error(f"LLM query error: {e}", exc_info=True)
            return ""

    def _process_objective_functions(self, network_instance: Thing) -> None:
        """
        Process loss and regularizer functions, and link them to it's network instance.
        """
        if not isinstance(network_instance, Thing):
            self.logger.error(
                "Network instance not provided in process_objective_functions."
            )
            raise TypeError(
                "Expected an instance of 'Thing' for Network in _process_objective_function."
            )

        if not hasattr(self.ontology, "ObjectiveFunction") or not isinstance(
            self.ontology.ObjectiveFunction, ThingClass
        ):
            self.logger.error("ObjectiveFunction class not found in ontology.")
            raise AttributeError(
                "The ontology must have a valid ObjectiveFunction class of type ThingClass."
            )

        # TODO: Assumes a network has only one loss function and regularizer function.

        try:

            # Get the name of the network instance
            network_instance_name = self._unhash_and_format_instance_name(
                network_instance.name
            )

            # TODO: Find better place to put this
            general_network_header_prompt = "You are an expert in neural network architectures with deep knowledge of various models, including CNNs, RNNs, Transformers, and other advanced architectures. Your goal is to extract and provide accurate, detailed, and context-specific information about a given neural network architecture from the provided context.\n\n"
            objective_function_prompt = f"Extract the loss function and regularizer function details used in only the {network_instance_name}."
            objective_function_prompt = (
                general_network_header_prompt + objective_function_prompt
            )  # TEMP

            objective_function_json_format_prompt = (
                f"- loss function: a string representing the type of loss function used in the {network_instance_name} network.\n"
                "- regularizer function: a string representing the type of regularizer function used in along with the loss function.\n"
                "- objective function: a string representing whether the loss function function is set to 'minimize' or 'maximize', where minimization reduces prediction errors (e.g., in regression and classification tasks) and maximization enhances desired outcomes (e.g., in reinforcement learning or adversarial training).\n\n"
                "For example, if the loss function is 'Mean Squared Error', the regularizer function is 'L1', and the objective function is set to 'maximize', the output should look like:\n"
                "{\n"
                '"answer": {\n'
                '"cost_function": {\n'
                '"lossFunction": "Mean Squared Error",\n'
                '"regularFunction": "L1"\n'
                "},\n"
                '"objectiveFunction": "maximize"\n'
                "}\n"
                "}\n"
                "If the regularizer function is not available, you may return None.\n"
            )

            objective_function_response = self._query_llm(
                "",
                objective_function_prompt,
                objective_function_json_format_prompt,
                pydantic_type_schema=ObjectiveFunctionResponse,
            )

            if not objective_function_response:
                self.logger.warning(
                    f"No response for objective functions in network {network_instance_name}."
                )
                return

            # Extract the loss function and regularizer function details
            loss_function_name = str(
                objective_function_response.answer.cost_function.lossFunction
            )
            regularizer_function_name = str(
                objective_function_response.answer.cost_function.regularFunction
            )
            objective_function_type = str(
                objective_function_response.answer.objectiveFunction
            )

            # Instantiate the objective function based on the objective type
            if objective_function_type.lower() == "minimize":
                objective_function_instance = self._instantiate_and_format_class(
                    self.ontology.MinObjectiveFunction, "Min Objective Function"
                )
            elif objective_function_type.lower() == "maximize":
                objective_function_instance = self._instantiate_and_format_class(
                    self.ontology.MaxObjectiveFunction, f"Max Objective Function"
                )
            else:
                self.logger.warning(
                    f"Invalid response for loss function objective type for {loss_function_name}, using minimzie as default."
                )
                objective_function_instance = self._instantiate_and_format_class(
                    self.ontology.MinObjectiveFunction, "Min Objective Function"
                )  # Default to minimize if no response
            
            # Link objective function instance to network instance.
            self._link_instances(
                network_instance,
                objective_function_instance,
                self.ontology.hasObjectiveFunction,
            )

            # Get all known loss functions for the loss function
            known_loss_functions = get_all_subclasses(self.ontology.LossFunction)

            if not known_loss_functions:
                self.logger.warning(
                    f"No known loss functions found in the ontology, created subclass for {loss_function_name} in the {network_instance_name}."
                )
                best_match_loss_class = create_subclass(
                    self.ontology, loss_function_name, self.ontology.LossFunction
                )
            else:
                # Check if the loss function name matches any known loss function
                best_match_loss_class = self._fuzzy_match_class(
                    loss_function_name, known_loss_functions, 90
                )
                if not best_match_loss_class:
                    best_match_loss_class = create_subclass(
                        self.ontology, loss_function_name, self.ontology.LossFunction
                    )

            # Instantiate the cost function and loss function
            cost_function_instance = self._instantiate_and_format_class(
                self.ontology.CostFunction, "cost function"
            )
            loss_function_instance = self._instantiate_and_format_class(
                best_match_loss_class, loss_function_name
            )

            # Link the objective function to the cost function and the cost function to the loss function
            self._link_instances(
                objective_function_instance,
                cost_function_instance,
                self.ontology.hasCost,
            )
            self._link_instances(
                cost_function_instance,
                loss_function_instance,
                self.ontology.hasLoss,
            )

            # Instantiate the regularizer function if provided
            if regularizer_function_name:
                best_match_reg_class = self._fuzzy_match_class(
                    regularizer_function_name, known_loss_functions, 90
                )

                if not best_match_reg_class:
                    best_match_reg_class = create_subclass(
                        self.ontology,
                        regularizer_function_name,
                        self.ontology.RegularizerFunction,
                    )

                reg_instance = self._instantiate_and_format_class(
                    best_match_reg_class, regularizer_function_name
                )
                self._link_instances(
                    cost_function_instance,
                    reg_instance,
                    self.ontology.hasRegularizer,
                )

            self.logger.info(
                f"Processed objective functions for {network_instance_name}: Loss Function: {loss_function_name}, Regularizer Function: {regularizer_function_name}, Objective Function Type: {objective_function_type}."
            )

        except Exception as e:
            self.logger.error(
                f"Error processing objective functions: {e}", exc_info=True
            )

    def _extract_network_data(self) -> list:
        """
        Extract name & type for each layer found within relevant parsed code

        :return List of dictionaries containing name & type per layer
        """
        network_json_path = glob.glob(f"data/{self.ann_config_name}/*.json")
        extracted_data: list = []

        for file in network_json_path:
            with open(file , "r") as f:
                try:
                    data: dict = json.load(f)

                    if 'network' in data and isinstance(data['network'] , list):
                        for layer in data['network']:
                            if 'name' in layer and 'type' in layer and layer['type'] is not None:
                                extracted_data.append({'name': layer['name'] , 'type': layer['type']})
                except Exception as e:
                    self.logger.exception(f"Error extracting JSON network data in {file} {e}" , exc_info=True)
        
        return extracted_data
    
    def _process_parsed_code(self , network_instance: Thing) -> None:
        """
        Process code that has been parsed into specific JSON structure to instantiate
        an ontology for the network associated with the code

        NOTE: cleanup & modularization needed w db layer processing

        :param network_instance: the network instance
        :return None
        """
        try:
            network_data: dict = self._extract_network_data()
            if network_data is None:
                warnings.warn("No parsed code available for given network")

            layer_subclasses: list = get_all_subclasses(self.ontology.Layer)
            actfunc_subclasses: list = get_all_subclasses(self.ontology.ActivationFunction)
            layer_subclasses.extend(actfunc_subclasses)

            for layer in network_data:
                layer_name = layer['name']
                layer_type = layer['type']

                best_layer_match = self._fuzzy_match_class(layer_type , layer_subclasses , 70)
                #best_actfunc_match = self._fuzzy_match_class(layer_type , actfunc_subclasses , 70)

                if not best_layer_match: # create subclass if layer type not found
                    best_layer_match = create_subclass(self.ontology , layer_type , self.ontology.Layer)
                    layer_subclasses.append(best_layer_match)

                # deal with activation functions later (how do we know uninstantiated layer is an activation layer?)

                layer_instance = self._instantiate_and_format_class(best_layer_match , layer_name)
                self._link_instances(network_instance , layer_instance , self.ontology.hasLayer)

            self.logger.info(f"All layers of {self.ann_config_name} processed")
        except Exception as e:
            self.logger.error(f"Error processing parsed code {e}" , exc_info=True)

    def _process_layers(self, network_instance: Thing) -> None:
        """
        Process the different layers (input, output, activation, noise, and modification) of it's network instance.

        :param network_instance: the network instance
        :return None
        """
        try:
            # fetch info from database
            onn = OnnxAddition()
            onn.init_engine()
            models_list = onn.fetch_models()
            num_models = len(models_list)
            prev_model = None
            subclasses:List[ThingClass] = get_all_subclasses(self.ontology.Layer)

            #### NOTE
            # accounts for undetailed ontology wherein activation functions
            # are treated as layers (prevents duplication)
            subclasses.extend(get_all_subclasses(self.ontology.ActivationFunction))

            # fetch ann config name, find relevant model in database
            best_model_name = self._fuzzy_match_list(models_list)
            if not best_model_name:
                warnings.warn(f"Model name {best_model_name} not found in database")
                self._llm_process_layers(network_instance) # for now...
                # throw to josue's script for llm instantiation?

            # fetch layer list of relevant model
            layer_list = onn.fetch_layers(best_model_name)

            for name in layer_list:
                layer_name , layer_type , model_id , model_name = name

                # Trying to be robust to weird db situations
                if layer_name is None:
                    continue
                if layer_name.lower() == self.ann_config_name.lower():
                    continue

                best_subclass_match = self._fuzzy_match_class(layer_type , subclasses , 70)
                if not best_subclass_match: # create subclass if layer type not found in ontology
                    best_subclass_match = create_subclass(self.ontology , layer_type , self.ontology.Layer)
                    subclasses.append(best_subclass_match) # track subclasses, ensure no duplicates
                
                # Debugging
                if model_id != prev_model:
                    self.logger.info(f"Processing model {model_id} / {num_models}")
                    self.logger.info(f"Model name {model_name}, subclass match {best_subclass_match}")

                layer_instance = self._instantiate_and_format_class(best_subclass_match , layer_name)
                self._link_instances(network_instance , layer_instance , self.ontology.hasLayer)

            self.logger.info(f"All layers of {model_name} successfully processed")

        except Exception as e:
            print("ERROR")
            self.logger.error(f"Error in _process_layers: {e}",exc_info=True)

    def _llm_process_layers(self, network_instance: str) -> None:
        """
        Process the different layers (input, output, activation, noise, and modification) of it's network instance.
        """
        network_instance_name = self._unhash_and_format_instance_name(
            network_instance.name
        )

        # Process Input Layer
        input_layer_prompt = (
            f"Extract the input size information for the input layer of the {network_instance_name} architecture. "
            "If the network accepts image data, the input size will be specified by its dimensions in the format 'WidthxHeightxChannels' (e.g., '64x64x1', '128x128x3', '512x512x3'). "
            "In this case, return the input dimensions as a string exactly in that format. "
            "If the network is not image-based, determine the total number of input units (neurons or nodes) and return that number as an integer. "
            "Your answer must be provided in JSON format with the key 'answer'.\n\n"
            "Examples:\n"
            "1. For an image-based network (e.g., a SVM) with input dimensions of 128x128x3:\n"
            '{"answer": "128x128x3"}\n\n'
            "2. For a network (e.g., a Generator) that takes a 100-dimensional vector as input:\n"
            '{"answer": 100}\n\n'
            "3. For a network (e.g., a Linear Regression model) with a single input feature:\n"
            '{"answer": 1}\n\n'
            f"Now, for the following network:\nNetwork: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )

        input_units = self._query_llm("", input_layer_prompt)
        if not input_units:
            self.logger.info("No response for input layer units.")
        else:
            input_layer_instance = self._instantiate_and_format_class(
                self.ontology.InputLayer, "Input Layer"
            )
            input_layer_instance.layer_num_units = [input_units]
            self._link_instances(
                network_instance, input_layer_instance, self.ontology.hasLayer
            )

        # Process Output Layer
        output_layer_prompt = (
            f"Extract the number of units in the output layer of the {network_instance_name} architecture. "
            "The number of units refers to the number of neurons or nodes in the output layer. "
            "Return the result as an integer in JSON format with the key 'answer'.\n\n"
            "Examples:\n"
            "1. Network: Discriminator\n"
            '{"answer": 1}\n\n'
            "2. Network: Generator\n"
            '{"answer": 784}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": 1}\n\n'
            f"Now, for the following network:\nNetwork: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        output_units = self._query_llm("", output_layer_prompt)
        if not output_units:
            self.logger.info("No response for output layer units.")
        else:
            output_layer_instance = self._instantiate_and_format_class(
                self.ontology.OutputLayer, "Output Layer"
            )
            output_layer_instance.layer_num_units = [output_units]
            self._link_instances(
                network_instance, output_layer_instance, self.ontology.hasLayer
            )

        # Process Activation Layers
        activation_layer_prompt = (
            f"Extract the number of instances of each core layer type in the {network_instance_name} architecture. "
            "Only count layers that represent essential network operations such as convolutional layers, "
            "fully connected (dense) layers, and attention layers.\n"
            "Do NOT count layers that serve as noise layers (i.e. guassian, normal, etc), "
            "activation functions (e.g., ReLU, Sigmoid), or modification layers (e.g., dropout, batch normalization), "
            "or pooling layers (e.g. max pool, average pool).\n\n"
            'Please provide the output in JSON format using the key "answer", where the value is a dictionary '
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
            '  "answer": {\n'
            '    "Convolutional": 3,\n'
            '    "Fully Connected": 2,\n'
            '    "Recurrent": 2,\n'
            '    "Attention": 1,\n'
            '    "Transformer Encoder": 3\n'
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
            '  "answer": {\n'
            '    "Convolutional": 4,\n'
            '    "FullyConnected": 1,\n'
            '    "Recurrent": 2,\n'
            '    "Attention": 1,\n'
            '    "Transformer Encoder": 3\n'
            "  }\n"
            "}\n\n"
            "Now, for the following network:\n"
            f"Network: {network_instance_name}\n"
            "Expected JSON Output:\n"
            "{\n"
            '  "answer": "<Your Answer Here>"\n'
            "}\n"
        )
        activation_layer_counts = self._query_llm("", activation_layer_prompt)
        if not activation_layer_counts:
            self.logger.info("No response for activation layer classes.")
        else:
            for layer_type, layer_count in activation_layer_counts.items():
                for i in range(layer_count):
                    activation_layer_instance = self._instantiate_and_format_class(
                        self.ontology.ActivationLayer, f"{layer_type} {i + 1}"
                    )
                    activation_layer_instance_name = (
                        self._unhash_and_format_instance_name(
                            activation_layer_instance.name
                        )
                    )
                    self._link_instances(
                        network_instance,
                        activation_layer_instance,
                        self.ontology.hasLayer,
                    )
                    # Process bias for activation layer
                    layer_ordinal = int_to_ordinal(i + 1)
                    bias_prompt = (
                        f"Does the {layer_ordinal} {activation_layer_instance_name} layer include a bias term? "
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
                        f"Now, for the following layer:\nLayer: {layer_ordinal} {activation_layer_instance_name}\n"
                        '{"answer": "<Your Answer Here>"}'
                    )
                    has_bias_response = self._query_llm("", bias_prompt)
                    if has_bias_response:
                        if has_bias_response.lower() == "true":
                            activation_layer_instance.has_bias = [True]
                        elif has_bias_response.lower() == "false":
                            activation_layer_instance.has_bias = [False]
                        self.logger.info(
                            f"Set bias term for {layer_ordinal} {activation_layer_instance_name} to {activation_layer_instance.has_bias}."
                        )

                    # Process activation function for activation layer
                    activation_function_prompt = (
                        f"Goal:\nIdentify the activation function used in the {layer_ordinal} {activation_layer_instance_name} layer, if any.\n\n"
                        "Return Format:\nRespond with the activation function name in JSON format using the key 'answer'. If there is no activation function or it's unknown, return an empty list [].\n"
                        "Examples:\n"
                        '{"answer": "ReLU"}\n'
                        '{"answer": "Sigmoid"}\n'
                        '{"answer": []}\n\n'
                        f"Now, for the following layer:\nLayer: {layer_ordinal} {activation_layer_instance_name}\n"
                        '{"answer": "<Your Answer Here>"}'
                    )
                    activation_function_response = self._query_llm(
                        "", activation_function_prompt
                    )
                    if activation_function_response:
                        if activation_function_response != "[]":
                            activation_function_instance = (
                                self._instantiate_and_format_class(
                                    self.ontology.ActivationFunction,
                                    activation_function_response,
                                )
                            )
                            self._link_instances(
                                activation_layer_instance,
                                activation_function_instance,
                                self.ontology.hasActivationFunction,
                            )
                        else:
                            self.logger.info(
                                f"No activation function associated with {layer_ordinal} {activation_layer_instance_name}."
                            )

        # Process Noise Layers
        noise_layer_prompt = (
            f"Does the {network_instance_name} architecture include any noise layers? "
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
            f"Now, for the following network:\nNetwork: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        noise_layer_response = self._query_llm("", noise_layer_prompt)
        if not noise_layer_response:
            self.logger.info("No response for noise layer classes.")
        elif noise_layer_response.lower() == "true":
            noise_layer_pdf_prompt = (
                f"Extract the probability distribution function (PDF) and its associated hyperparameters for the noise layers in the {network_instance_name} architecture. "
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
                f"Now, for the following network:\nNetwork: {network_instance_name}\n"
                '{"answer": "<Your Answer Here>"}'
            )
            noise_layer_pdf = self._query_llm("", noise_layer_pdf_prompt)
            if not noise_layer_pdf:
                self.logger.info("No response for noise layer PDF.")
            else:
                try:
                    if isinstance(noise_layer_pdf, dict):
                        for (
                            noise_name,
                            noise_params,
                        ) in (
                            noise_layer_pdf.items()
                        ):  # Not sure if this is the correct way to iterate over the dictionary.
                            noise_layer_instance = self._instantiate_and_format_class(
                                self.ontology.NoiseLayer, noise_name
                            )
                            self._link_instances(
                                network_instance,
                                noise_layer_instance,
                                self.ontology.hasLayer,
                            )
                            for (
                                param_name,
                                param_value,
                            ) in (
                                noise_params.items()
                            ):  # Not sure if this is the correct way to assign unknown data properties, filler for now.
                                setattr(noise_layer_instance, param_name, [param_value])
                except Exception as e:
                    self.logger.error(f"Error processing noise layer: {e}")

            # Process Modification Layers
            modification_layer_prompt = (
                f"Extract the number of instances of each modification layer type in the {network_instance_name} architecture. "
                "Modification layers include layers that alter the input data or introduce noise, such as Dropout, Batch Normalization, and Layer Normalization. "
                "Exclude noise layers (e.g., Gaussian Noise, Dropout) and activation layers (e.g., ReLU, Sigmoid) from your count.\n"
                'Please provide the output in JSON format using the key "answer", where the value is a dictionary '
                "mapping the layer type names to their counts.\n\n"
                "Examples:\n\n"
                "1. Network Architecture Description:\n"
                "- 3 Dropout layers\n"
                "- 2 Batch Normalization layers\n"
                "- 1 Layer Normalization layer\n"
                "Expected JSON Output:\n"
                "{\n"
                '  "answer": {\n'
                '    "Dropout": 3,\n'
                '    "Batch Normalization": 2,\n'
                '    "Layer Normalization": 1\n'
                "  }\n"
                "}\n\n"
                "2. Network Architecture Description:\n"
                "- 3 Dropout layers\n"
                "- 2 Batch Normalization layers\n"
                "- 1 Layer Normalization layer\n"
                "Expected JSON Output:\n"
                "{\n"
                '  "answer": {\n'
                '    "Dropout": 3,\n'
                '    "Batch Normalization": 2,\n'
                '    "Layer Normalization": 1\n'
                "  }\n"
                "}\n\n"
                "Now, for the following network:\n"
                f"Network: {network_instance_name}\n"
                "Expected JSON Output:\n"
                "{\n"
                '  "answer": "<Your Answer Here>"\n'
                "}\n"
            )
            modification_layer_counts = self._query_llm("", modification_layer_prompt)
            if not modification_layer_counts:
                self.logger.info("No response for modification layer classes.")
            else:
                dropout_match = next(
                    (
                        s
                        for s in modification_layer_counts
                        if fuzz.token_set_ratio("dropout", s) >= 85
                    ),
                    None,
                )
                dropout_layer_rate = None
                if dropout_match:
                    dropout_rate_prompt = (
                        f"Extract the dropout rate for the Dropout layers in the {network_instance_name} architecture. "
                        "The dropout rate is the fraction of input units to drop during training. "
                        "Return the result as a float in JSON format with the key 'answer'.\n\n"
                        "Examples:\n"
                        "1. Network: Discriminator\n"
                        '{"answer": 0.5}\n\n'
                        "2. Network: Generator\n"
                        '{"answer": 0.3}\n\n'
                        "3. Network: Linear Regression\n"
                        '{"answer": 0.2}\n\n'
                        f"Now, for the following network:\nNetwork: {network_instance_name}\n"
                        '{"answer": "<Your Answer Here>"}'
                    )
                    dropout_layer_rate = self._query_llm("", dropout_rate_prompt)
                    if not dropout_layer_rate:
                        self.logger.info("No response for dropout layer rate.")
                for layer_type, layer_count in modification_layer_counts.items():
                    for i in range(layer_count):
                        if dropout_match and layer_type == dropout_match:
                            dropout_layer_instance = self._instantiate_and_format_class(
                                self.ontology.DropoutLayer, f"{layer_type} {i + 1}"
                            )
                            if dropout_layer_rate:
                                dropout_layer_instance.dropout_rate = [
                                    dropout_layer_rate
                                ]
                            self._link_instances(
                                network_instance,
                                dropout_layer_instance,
                                self.ontology.hasLayer,
                            )
                        else:
                            modification_layer_instance = (
                                self._instantiate_and_format_class(
                                    self.ontology.ModificationLayer,
                                    f"{layer_type} {i + 1}",
                                )
                            )
                            self._link_instances(
                                network_instance,
                                modification_layer_instance,
                                self.ontology.hasLayer,
                            )

    def _process_task_characterization(self, network_instance: Thing) -> None:
        """
        Process the task characterization of a network instance.
        """
        try:
            if not isinstance(network_instance, Thing):
                self.logger.error("Expected an instance of Thing for Network in _process_task_characterization.")
                raise ValueError("Expected an instance of Thing for Network in _process_task_characterization.")
            if not hasattr(self.ontology, "TaskCharacterization") or not isinstance(self.ontology.TaskCharacterization, ThingClass):
                self.logger.error("The ontology must have a valid TaskCharacterization class of type ThingClass..")

            # TODO: Dynmaically provide tasks in prompt considering known task types.
            # TODO: Assumes only ones task per network, may need to change to multiple tasks.

            # Get the name of the network instance
            network_instance_name = self._unhash_and_format_instance_name(
                network_instance.name
            )

            # TODO: Find better place to put this
            general_network_header_prompt = (
                    "You are an expert in neural network architectures with deep knowledge of various models, including CNNs, RNNs, Transformers, and other advanced architectures. Your goal is to extract and provide accurate, detailed, and context-specific information about a given neural network architecture from the provided context.\n\n"
                )
            task_characterization_prompt = (
                f"Extract the primary task that the {network_instance_name} is designed to perform. "
            )
            task_characterization_prompt = general_network_header_prompt + task_characterization_prompt # TEMP

            task_characterization_json_format_prompt = (
                "The primary task is the most important or central objective of the network. "
                "Return the task name in JSON format with the key 'answer'.\n\n"
                # "Examples of types of tasks include:\n"
                # "- **Adversarial**: The task of generating adversarial examples or countering another network’s predictions, often used in adversarial training or GANs. \n"
                # "  Example: A model that generates images to fool a classifier.\n\n"
                # "- **Self-Supervised Classification**: The task of learning useful representations without explicit labels, often using contrastive or predictive learning techniques. \n"
                # "  Example: A network pre-trained using contrastive learning and later fine-tuned for classification.\n\n"
                # "- **Semi-Supervised Classification**: A classification task where the network is trained on a mix of labeled and unlabeled data. \n"
                # "  Example: A model trained with a small set of labeled images and a large set of unlabeled ones for better generalization.\n\n"
                # "- **Supervised Classification**: The task of assigning input data to predefined categories using fully labeled data. \n"
                # "  Example: A CNN trained on labeled medical images to classify diseases.\n\n"
                # "- **Unsupervised Classification (Clustering)**: The task of grouping similar data points into clusters without predefined labels. \n"
                # "  Example: A model that clusters news articles into topics based on similarity.\n\n"
                # "- **Discrimination**: The task of distinguishing between different types of data distributions, often used in adversarial training. \n"
                # "  Example: A discriminator in a GAN that differentiates between real and generated images.\n\n"
                # "- **Generation**: The task of producing new data that resembles a given distribution. \n"
                # "  Example: A generative model that creates realistic human faces from random noise.\n\n"
                # "- **Reconstruction**: The task of reconstructing input data, often used in denoising or autoencoders. \n"
                # "  Example: A model that removes noise from images to restore the original content.\n\n"
                # "- **Regression**: The task of predicting continuous values rather than categorical labels. \n"
                # "  Example: A neural network that predicts house prices based on features like size and location.\n\n"
                # "If the network's primary task does not fit any of the above categories, provide a conciece description of the task instead using at maximum a few words.\n\n"
                "For example, if the network is designed to classify images of handwritten digits, the task would be 'Supervised Classification'.\n\n"
                "{\n"
                    '"answer": {\n'
                        '"task_type": "Supervised Classification"\n'
                    "}\n"
                "}\n"
            )

            task_characterization_response = self._query_llm("", task_characterization_prompt, task_characterization_json_format_prompt, pydantic_type_schema=TaskCharacterizationResponse)

            if not task_characterization_response:
                self.logger.warning(f"No response for task characterization for network instance '{network_instance_name}'.")
                return
            
            # Extract the task type from the response
            task_type_name = str(task_characterization_response.answer.task_type)

            # Get all known task types for TaskCharacterization
            known_task_types = get_all_subclasses(self.ontology.TaskCharacterization)

            if not known_task_types:
                self.logger.warning(f"No known task types found in the ontology, creating a new task type for {task_type_name} in the {network_instance_name}.")
                best_match_task_type = create_subclass(self.ontology, task_type_name, self.ontology.TaskCharacterization)
            else:
                # Check if the task type matches any known task types
                best_match_task_type = self._fuzzy_match_class(task_type_name, known_task_types, 90)
                if not best_match_task_type:
                    best_match_task_type = create_subclass(self.ontology, task_type_name, self.ontology.TaskCharacterization)

            # Instantiate and link the task characterization instance with the network instance
            task_type_instance = self._instantiate_and_format_class(best_match_task_type, task_type_name)
            self.logger.info(f"Processed task characterization '{task_type_name}', linked to network instance '{network_instance_name}.")

            self._link_instances(network_instance, task_type_instance, self.ontology.hasTaskType)
        except Exception as e:
            self.logger.error(f"Error processing task characerization for network instance '{network_instance_name}': {e}",exc_info=True)

    def _process_network(self, ann_config_instance: Thing) -> None:
        """
        Process the network class and it's components.
        """
        try:
            if not isinstance(ann_config_instance, Thing):
                logger.error("Invalid ANN Configuration instance.")
                raise TypeError(
                    "Expected an instance of 'Thing' for ANN Configuration in _process_network."
                )

            if not hasattr(self.ontology, "Network") or not isinstance(
                self.ontology.Network, ThingClass
            ):
                logger.error("Invalid or missing Network class in the ontology.")
                raise AttributeError(
                    "The ontology must have a valid 'Network' class of type ThingClass."
                )
            network_instances: List[ThingClass] = []  # List of network instances

            # Here is where logic for processing the network instance would go.
            network_instances.append(
                self._instantiate_and_format_class(
                    self.ontology.Network, "Convolutional Network"
                )
            )  # assumes network is convolutional for cnn

            # Process the components of the network instance.
            for network_instance in network_instances:
                # Link the network instance to the ANN Configuration instance,
                self._link_instances(
                    ann_config_instance, network_instance, self.ontology.hasNetwork
                )
                self._process_layers(network_instance) # May be processed by onnx
                #self._process_objective_functions(network_instance)
                #self._process_task_characterization(network_instance)
        except Exception as e:
            self.logger.error(
                f"Error processing the '{ann_config_instance}' networks: {e}",
                exc_info=True,
            )
            raise e

    def __addclasses(self) -> None:
        """Adds new predefined classes to the ontology."""
        new_classes = {
            "Self-Supervised Classification": self.ontology.TaskCharacterization,
            "Unsupervised Classification": self.ontology.TaskCharacterization,
        }

        for name, parent in new_classes.items():
            try:
                create_subclass(self.ontology, name, parent)
            except Exception as e:
                self.logger.error(
                    f"Error creating new class {name}: {e}", exc_info=True
                )

    def save_ontology(self) -> None:
        """
        Saves the ontology to the pre-specified file.
        """
        self.ontology.save(file=self.output_owl_path, format="rdfxml")
        self.logger.info(f"Ontology saved to {self.output_owl_path}")

    def run(self) -> None:
        """
        Main method to run the ontology instantiation process.
        """
        try:
            with self.ontology:
                start_time = time.time()

                if not hasattr(self.ontology, "ANNConfiguration"):
                    raise AttributeError(
                        "Error: Class 'ANNConfiguration' not found in ontology."
                    )

                # Initialize the LLM engine for each json_document context in paper and/or code.
                for count, j in enumerate(self.list_json_doc_paths):
                    init_engine(self.ann_config_name, j)

                self.__addclasses()  # Add new general classes to ontology #TODO: better logic for doing this elsewhere

                # Instantiate the ANN Configuration class.
                ann_config_instance = self._instantiate_and_format_class(
                    self.ontology.ANNConfiguration, self.ann_config_name
                )

                # Process the network class and it's components.
                #self._process_network(ann_config_instance)
                self._process_parsed_code(ann_config_instance) # network name?

                # Process TrainingStrategy and it's components.
                # self._process_training_strategy(ann_config_instance)

                # Log time taken to instantiate the ANN ontology instance.
                minutes, seconds = divmod(time.time() - start_time, 60)
                logging.info(
                    f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds."
                )

                logging.info(
                    f"Ontology instantiation completed for {self.ann_config_name}."
                )

        except Exception as e:
            self.logger.error(
                f"Error during the {self.ann_config_name} ontology instantiation: {e}",
                exc_info=True,
            )
            raise e

def test_run_layers():
    pass


# For standalone testing
if __name__ == "__main__":
    import glob

    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"

    for model_name in [
        "alexnet", # id = 191
        # "resnet", # id = 198
        # "vgg16", # id = 206
        #"gan", # Assume we can model name from user or something
    ]:
        try:
            code_files = glob.glob(f"data/{model_name}/*.py")
            pdf_file = f"data/{model_name}/{model_name}.pdf"

            # Here these paths will each need to be extracted from the PDF and code files to json_docs.json

            # Now we have JSON files for both the papers and code files respectively.
            list_json_doc_paths = glob.glob(f"data/{model_name}/*.json")

            instantiator = OntologyInstantiator(
                ontology_path, list_json_doc_paths, model_name
            )
            instantiator.run()
            instantiator.save_ontology()
        except Exception as e:
            print(f"Error instantiating the {model_name} ontology in __name__: {e}")
            continue