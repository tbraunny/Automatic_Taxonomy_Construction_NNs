import hashlib
import os
import time
import json
import glob
from typing import Dict, Any, Union, List, Optional
from tests.deprecated.onnx_db import OnnxAddition

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
    create_class_data_property,
    link_data_property_to_instance,
    create_class_object_property,
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

# Initialize logger
logger = get_logger("instantiate_annetto")

class OntologyInstantiator:
    """
    A class to instantiate an annett-o ontology by processing each main component separately and linking them together.
    """

    def __init__(
        self,
        list_json_doc_paths: List[str],
        ann_config_name: str = "alexnet",
        output_owl_path: str = "data/annett-o-test.owl",
    ) -> None:
        """
        Initialize the OntologyInstantiator class.
        # Args:
            json_doc_files_paths (list[str]): The list of str paths to the JSON_doc files for paper and/or code.
            ann_config_name (str): The name of the ANN configuration.
            ontology (str): The ontology. 
            ontology_output_filepath (str): The .owl path to save the ontology file.
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
        self.ann_config_name = ann_config_name.lower().strip()
        self.output_owl_path = ontology_output_filepath

        self.llm_cache: Dict[str, Any] = {}
        self.logger = logger
        self.ann_config_hash = self._generate_hash(self.ann_config_name)

    def _generate_hash(self, str: str) -> str:
        """
        Generate a unique hash identifier based on the given string.
        """
        hash_object = hashlib.md5(str.encode())  # Generate a consistent hash
        return hash_object.hexdigest()[:8]

    def _instantiate_and_format_class(
        self, cls: ThingClass, instance_name: str, source: Optional[str] = None
    ) -> Thing:
        """
        Instantiate a given ontology class with the specified instance name.
        Uses the ANN configuration hash as a prefix for uniqueness.
        :param cls: The ontology class to instantiate.
        :param instance_name: The name of the instance to create.
        :param source: Optional source for the instance (i.e. 'code' or 'paper').
        :return: The instantiated Thing object.
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
        stripped_name = stripped_name.replace("-", " ")  # Convert dashes back to spaces
        # capitalize the first letter of each word
        stripped_name = " ".join(word.capitalize() for word in stripped_name.split())
        return stripped_name

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
            raise TypeError("Expected instance_name to be a string.", exc_info=True)
        if not all(isinstance(cls, ThingClass) for cls in classes):
            raise TypeError(
                "Expected classes to be a list of ThingClass objects.", exc_info=True
            )
        if not all(isinstance(cls.name, str) for cls in classes):
            raise TypeError(
                "Expected classes to have string names. ######", exc_info=True
            )
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.", exc_info=True)

        # Convert classes to a dictionary for lookup
        class_name_map = {cls.name.lower(): cls for cls in classes}

        match, score, _ = process.extractOne(
            instance_name.lower(), class_name_map.keys(), scorer=fuzz.ratio
        )
        # might need to reupper names later capitalized_string = string[0].upper() + string[1:]

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
            raise TypeError(
                "Expected class_names to be a list of strings.", exc_info=True
            )
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.")
        
        class_names_lower = [name.lower() for name in class_names]
        match, score, _ = process.extractOne(self.ann_config_name.lower(), class_names_lower, scorer=fuzz.ratio)

        return match if score >= threshold else None

    def _link_instances(
        self,
        parent_instance: Thing,
        child_instance: Thing,
        object_property: ObjectPropertyClass,
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

    def _link_data_property(
        self, instance: Thing, data_property: DataPropertyClass, value: Any
    ) -> None:
        """
        Link a data property to an instance.
        """
        link_data_property_to_instance(instance, data_property, value)
        self.logger.info(
            f"Linked '{self._unformat_instance_name(instance.name)}' with data property '{data_property.name}'."
        )
    
    def _add_source_data_property(self, instance: Thing, source: str) -> None:
        """
        Add a source data property to an instance.
        """
        try:
            source_property = create_class_data_property(
                self.ontology, "source", type(instance), str, False
            )
        except Exception as e:
            self.logger.warning(
                f"Error creating source data property: {e}", exc_info=True
            )
        link_data_property_to_instance(instance, source_property, source)
        self.logger.info(
            f"Linked '{self._unformat_instance_name(instance.name)}' with source data property."
        )

    def _add_definition_data_property(self, instance: Thing, definition: str) -> None:

        try:
            definition_property = create_class_data_property(
                self.ontology, "definition", type(instance), str, False
            )
        except Exception as e:
            self.logger.warning(
                f"Error creating definition data property: {e}", exc_info=True
            )
        link_data_property_to_instance(instance, definition_property, definition)
        self.logger.info(
            f"Linked '{self._unformat_instance_name(instance.name)}' with definition data property."
        )

    def build_prompt(
        self,
        task: str,
        query: str,
        instructions: str,
        examples: str,
        extra_instructions: str = "",
    ) -> str:
        return f"{task}\n{instructions}\n{examples}\n{extra_instructions}\nNow, for the following:\n{query}\n "

    def _query_llm(
        self,
        prompt: str,
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
        if prompt in self.llm_cache:
            self.logger.info(f"Using cached LLM response for prompt: {prompt}")
            print("Using cached LLM response WOOT #####")
            return self.llm_cache[prompt]
        try:
            # Response returned as pydantic class if json_format_instructions and pydantic_type_schema are provided.
            response = query_llm(
                self.ann_config_name,
                prompt,
                pydantic_type_schema,
                max_chunks=20,
                token_budget=5000
            )
            self.llm_cache[prompt] = response

            return response
        except Exception as e:
            self.logger.error(f"LLM query error: {e}", exc_info=True)
            return ""

    def _process_objective_functions(self, network_instance: Thing) -> None:
        network_name = self._unformat_instance_name(network_instance.name)

        # Define examples using defintions
        examples = (
            "Examples:\n"
            "Network: Discriminator\n"
            """{"answer": {
                "loss": {
                    "name": "Power-Outlet Loss",
                    "definition": "Measures energy imbalance between predicted and real outputs."
                },
                "regularizer": {
                    "name": "Electric Regularization",
                    "definition": "Penalizes current surges to stabilize the model."
                },
                "objective": "minimize"
            }}\n\n"""
            "Network: Generator\n"
            """{"answer": {
                "loss": {
                    "name": "Power-Outlet Loss",
                    "definition": "Measures energy imbalance between predicted and real outputs."
                },
                "regularizer": null,
                "objective": "maximize"
            }}\n"""
        )

        task = "Extract the loss function, regularizer, and objective type for a network.\n"
        instructions = (
            "Return the response in JSON format with the key 'answer'.\n"
            "If the loss function or regularizer is not explicitly named, infer the most likely standard name based on its description (e.g., 'maximize the log-likelihood of correct class' may correspond to a well-known loss function).\n"
        )
        query = f"Network: {network_name}"
        extra_instructions = (
            "Objective type must be 'minimize' or 'maximize'. "
            "Regularizer can be null if not specified."
        )

        prompt = self.build_prompt(
            task, query, instructions, examples, extra_instructions
        )

        response = self._query_llm(prompt, ObjectiveFunctionResponse)
        if not response:
            self.logger.warning(
                f"No response for objective functions in network {network_name}."
            )
            return
        
        loss_name = get_sanitized_attr(response, "loss.name")
        loss_def = get_sanitized_attr(response, "loss.definition")
        reg_name = get_sanitized_attr(response, "regularizer.name")
        reg_def = get_sanitized_attr(response, "regularizer.definition")
        obj_type = get_sanitized_attr(response, "objective")
        
        # loss_name = str(response.loss.name)
        # loss_def = str(response.loss.definition)

        # Objective function handling
        if obj_type:
            obj_cls = (
                self.ontology.MinObjectiveFunction
                if obj_type.lower().strip() == "minimize"
                else self.ontology.MaxObjectiveFunction
            )
            obj_instance = self._instantiate_and_format_class(
                obj_cls, f"{obj_type} Objective Function"
            )

        # TODO: Assumes a network has only one loss function and regularizer function.

        try:

            # Get the name of the network instance
            network_instance_name = self._unhash_and_format_instance_name(
                network_instance.name
            )
            self.logger.warning(f"No objective type specified for {network_name}. Defaulting to MinObjectiveFunction.")


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
            loss_instance = self._instantiate_and_format_class(best_loss_match, loss_name)
            self._link_instances(obj_instance, cost_instance, self.ontology.hasCost)
            self._link_instances(cost_instance, loss_instance, self.ontology.hasLoss)
        
            # Add definition to loss instance
            if loss_def:
                self._add_definition_data_property(loss_instance, loss_def)

        # Regularizer handling
        if reg_name:
            best_reg_match = self._fuzzy_match_class(
                reg_name, known_losses, 90
            ) or create_subclass(
                self.ontology, reg_name, self.ontology.RegularizerFunction
            )
            reg_instance = self._instantiate_and_format_class(best_reg_match, reg_name)
            self._link_instances(
                cost_instance, reg_instance, self.ontology.hasRegularizer
            )
            if reg_def:
                self._add_definition_data_property(reg_instance, reg_def)

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
        DEPRECATED
        Extract relevant model data from parsed JSON's

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

        NOTE: modularize w onnx db parsing
        - include parameter counts per layer (allow to compute model total)
        - insert new models into the onnx database? (still somewhat unsure about this, how to best accomplish it)

        :param network_instance: the network instance
        :return None
        """
        # list of activation functions (append to as needed)
        activation_functions = [
            "Softmax",
            "ReLU",
            "Tanh",
            "Linear",
            "GAN_Generator_Tanh",
            "GAN_Generator_ReLU",
            "GAN_Discriminator_Sigmoid",
            "GAN_Discriminator_ReLU",
            "AAE_Encoder_Linear",
            "AAE_Encoder_ReLU",
            "AAE_Encoder_Softmax",
            "AAE_Encoder_ZClone_ReLU",
            "AAE_Encoder_YClone_ReLU",
            "AAE_Decoder_Sigmoid",
            "AAE_Decoder_ReLU",
            "AAE_Style_Discriminator_ReLU",
            "AAE_Style_Discriminator_Sigmoid",
            "AAE_Label_Discriminator_Sigmoid",
            "AAE_Label_Discriminator_ReLU",
            "ELU",
            "Hardshrink",
            "Hardsigmoid",
            "Hardtanh",
            "Hardswish",
            "LeakyReLU",
            "LogSigmoid",
            "MultiheadAttention",
            "PReLU",
            "ReLU",
            "ReLU6",
            "RReLU",
            "SELU",
            "CELU",
            "GELU",
            "Sigmoid",
            "SiLU",
            "Mish",
            "Softplus",
            "Softshrink",
            "Softsign",
            "Tanh",
            "Tanhshrink",
            "Threshold",
            "GLU",
            # Non-Linear activations (others)
            "Softmin",
            "Softmax",
            "Softmax2d",
            "LogSoftmax",
            "AdaptiveLogSoftmaxWithLoss"
        ]

        json_file = glob.glob(f"data/**/{self.ann_config_name}.json") # grab current json
        network_data: dict = {}
        try:
            with open(json_file , 'r') as f:
                network_data = json.loads(f)

            nodes = network_data.get('graph' , {}).get('node' , {})
            #network_data: dict = self._extract_network_data()
            # if network_data is None:
            #     warnings.warn("No parsed code available for given network")

            layer_subclasses: list = get_all_subclasses(self.ontology.Layer)
            # NOTE: uncomment if method found for instantiating previously unknown activation functions
            #actfunc_subclasses: list = get_all_subclasses(self.ontology.ActivationFunction)
            #layer_subclasses.extend(actfunc_subclasses)

            name_to_instance: dict = {} # easier lookup between actfunc & layer, keeps instance in scope, less calls to ontology
            for layer in nodes:
                layer_name = layer.get('name')
                layer_type = layer.get('op_type')
                layer_params = layer.get('num_params')

                best_actfunc_match = self._fuzzy_match_class(layer_type , activation_functions , 70)
                #best_actfunc_match = self._fuzzy_match_class(layer_type , actfunc_subclasses , 70)

                if best_actfunc_match: # if activation function present
                    actfunc_instance = self._instantiate_and_format_class(best_actfunc_match , layer_name)
                    self._link_instances(network_instance , actfunc_instance , self.ontology.hasActivationFunction)
                    name_to_instance[layer_name] = actfunc_instance
                else: 
                    best_layer_match = self._fuzzy_match_class(layer_type , layer_subclasses , 70)
                    if not best_layer_match: # create subclass if layer type not found
                        best_layer_match = create_subclass(self.ontology , layer_type , self.ontology.Layer)
                        layer_subclasses.append(best_layer_match)

                    layer_instance = self._instantiate_and_format_class(best_layer_match , layer_name)
                    self._link_instances(network_instance , layer_instance , self.ontology.hasLayer)

                    # attach number of parameters to layer
                    if layer_params:
                        self._link_data_property(layer_instance , self.ontology.Layer.layer_num_units , layer_params)
                    else:
                        self.logger.info(f"Layer {layer_name} does not have a number of parameters.")
                    name_to_instance[layer_name] = layer_instance

            # second run for instantiating next, prev, and other linkages
            for layer in nodes:
                layer_name = layer.get('name')
                layer_type = layer.get('type')
                prev_layer = layer.get('input' , [])
                next_layer = layer.get('target' , [])

                layer_instance = name_to_instance.get(layer_name)
                if not layer_instance:
                    self.logger.error(f"Layer instance not found for {layer_name} , linkage unsuccessful")

                for prev in prev_layer: # link nextLayer & hasInputLayer
                    prev_layer_instance = name_to_instance.get(prev)
                    if prev_layer_instance:
                        self._link_instances(layer_instance , prev_layer_instance , self.ontology.previousLayer)
                        self._link_instances(network_instance , layer_instance , self.ontology.hasInputLayer)
                    else:
                        self.logger.error(f"Previous layer {prev} of {layer} not instantiated")
                
                for next in next_layer: # link prevLayer & hasOutputLayer
                    next_layer_instance = name_to_instance.get(next)
                    if next_layer_instance:
                        self._link_instances(layer_instance , next_layer_instance , self.ontology.nextLayer)
                        self._link_instances(network_instance , layer_instance , self.ontology.hasOutputLayer)
                    else:
                        self.logger.error(f"Next layer {next} of {layer} not instantiated")

            self.logger.info(f"All layers of {self.ann_config_name} processed")
        except Exception as e:
            self.logger.error(f"Error processing parsed code {e}" , exc_info=True)
    
    def _llm_process_layers(self, network_instance: Thing) -> None:
        self.logger.error("Process layers with LLM not implemented yet.")
        raise NotImplementedError("Process layers with LLM not implemented yet.")

    def _process_db_layers(self, network_instance: Thing) -> None:
        """
        DEPRECATED
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
        network_name = self._unformat_instance_name(network_instance.name)

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

            examples = (
                "Examples:\n"
                "Network: GAN\n"
                """{
                    "answer": {
                        "task_type": [
                            {
                                "name": "Adversarial",
                                "definition": "A training paradigm where two networks compete, typically a generator and a discriminator."
                            }
                        ]
                    }
                }\n\n"""
                "Network: GPT-2\n"
                """{
                    "answer": {
                        "task_type": [
                            {
                                "name": "Generation",
                                "definition": "The task of producing coherent output sequences, such as text, from learned representations."
                            }
                        ]
                    }
                }\n\n"""
                "Network: SimCLR\n"
                """{
                    "answer": {
                        "task_type": [
                            {
                                "name": "Self-Supervised Classification",
                                "definition": "A learning approach using pretext tasks to learn useful representations without labeled data."
                            }
                        ]
                    }
                }\n"""
            )

            extra_instructions = (
                "Choose the most appropriate task from this list:\n"
                "- Adversarial\n"
                "- Self-Supervised Classification\n"
                "- Semi-Supervised Classification\n"
                "- Supervised Classification\n"
                "- Unsupervised Classification\n"
                "- Discrimination\n"
                "- Generation\n"
                "- Clustering\n"
                "- Regression\n"
                # "If none of these apply, use a concise custom name using only one or a couple words."
            )

            prompt = self.build_prompt(
                task, query, instructions, examples, extra_instructions
            )

            response = self._query_llm(prompt, TaskCharacterizationResponse)
            if not response:
                self.logger.warning(
                    f"No response for task characterization in network {network_name}."
                )
                return
            
            task_type_name = get_sanitized_attr(response, "answer.task_type")
            task_type_def = get_sanitized_attr(response, "answer.task_type.definition")
            
            # task_type_name = str(response.answer.task_type.name)
            # task_type_def = str(response.answer.task_type.definition)

            if not task_type_name:
                self.logger.warning(
                    f"No task type name provided in task characterization for network {network_name}."
                )
                return
            
            # Task Handling
            known_task_types = get_all_subclasses(self.ontology.TaskCharacterization)
            if not known_task_types:
                self.logger.warning(
                    f"No known task types found in the ontology. Creating new task type '{task_type_name}'."
                )
                best_match_task_type = create_subclass(
                    self.ontology, task_type_name, self.ontology.TaskCharacterization
                )
            else:
                best_match_task_type = self._fuzzy_match_class(
                    task_type_name, known_task_types, 90
                ) or create_subclass(
                    self.ontology, task_type_name, self.ontology.TaskCharacterization
                )

            task_type_instance = self._instantiate_and_format_class(
                best_match_task_type, task_type_name
            )

            self._link_instances(
                network_instance, task_type_instance, self.ontology.hasTaskType
            )

            # Task definition handling
            if task_type_def:
                self._add_definition_data_property(
                    task_type_instance, task_type_def
                )

            self.logger.info(
                f"Processed task characterization for {network_name}: Task Type: {task_type_name}."
            )

        except Exception as e:
            self.logger.error(
                f"Error processing task characterization for network '{network_name}': {e}",
                exc_info=True,
            )

    def _process_network(self, ann_config_instance: Thing) -> None:
        """
        Process the network class and it's components.
        """
        try:
            if not isinstance(ann_config_instance, Thing):
                logger.error("Invalid ANN Configuration instance.", exc_info=True)
                raise TypeError(
                    "Expected an instance of 'Thing' for ANN Configuration in _process_network."
                )

            if not hasattr(self.ontology, "Network") or not isinstance(
                self.ontology.Network, ThingClass
            ):
                logger.error(
                    "Invalid or missing Network class in the ontology.", exc_info=True
                )
                raise AttributeError(
                    "The ontology must have a valid 'Network' class of type ThingClass."
                )
            ann_config_instance_name = self._unformat_instance_name(ann_config_instance.name)

            layers_parsed:bool = False # Flag for if layers have been parsed yet

            # # Process layers via code
            # parse_code_layers:bool = True # TODO: we need logic to determine if parsable code exist
            # if parse_code_layers:
            #     self._process_parsed_code(ann_config_instance)
            #     layers_parsed = True
            # ##############

            # Define examples using defintions
            examples = (
            "Examples:\n"
            """{"answer": {
                "architectures": [
                    {
                        "architecture_name": "AlexNet",
                        "subnetworks": [
                            {"name": "convolutional network", "is_independent": true}
                        ]
                    }
                ]
            }}\n\n"""
            """{"answer": {
                "architectures": [
                    {
                        "architecture_name": "PulseFormer",
                        "subnetworks": [
                            {"name": "Pulse Encoder"},
                            {"name": "Temporal Filter"},
                            {"name": "Output Spike Layer"}
                        ]
                    },
                    {
                        "architecture_name": "FlowNet-Z",
                        "subnetworks": [
                            {"name": "Streamflow Predictor"},
                            {"name": "Hydrology Integration Module"}
                        ]
                    }
                ]
            }}\n\n"""
            "Network: GliderNet\n"
            """{"answer": {
                "architectures": [
                    {
                        "architecture_name": "GliderNet",
                        "subnetworks": [
                            {"name": "Wing Span Analysis Unit"}
                        ]
                    }
                ]
            }}\n"""
        )

            task = """You are a research assistant tasked with identifying neural network architectures and their components from academic papers. 
For the given paper, analyze the content carefully and precisely to extract the following:\n
Architecture Name(s): a named model or system composed of one or more subnetworks (e.g., “DeepONet”).\n
Subnetwork(s): a distinct functional block within an architecture that can be described independently (e.g., generator, encoder, trunk net).\n
A **subnetwork** is a block that\n
- Has its **own loss function**, or
- Is **trained or optimized independently**, or
- Is explicitly described in the paper as a separate module.\n
"""

            instructions = (
                "Return the response in JSON format with the key 'answer'.\n"
                "If a subnetwork is not named but has a described purpose, use a short functional label (e.g., “temporal integration module”).\n"
            )
            query = f""
            extra_instructions = (
                # f"Layers like convolution, pooling, or activation do not count as separate subnetworks unless they are grouped into a larger named module that is functionally distinct.\n"
                "Avoid listing low-level components like layers (e.g., convolution, fully-connected, pooling, activation) as separate subnetworks.\n"
                "If the architecture describes a single unified model composed of standard layers, return only one subnetwork with a high-level functional name (e.g., 'convolutional network' or 'feedforward network')."
            )

            prompt = self.build_prompt(
                task, query, instructions, examples, extra_instructions
            )

            response = self._query_llm(prompt, NetworkResponse)
            if not response or not response.answer.architectures:
                # self.logger.warning(f"No architectures found in network {network_name}.")
                return
            ann_config_instances: List[ThingClass] = []

            for architecture in response.answer.architectures:
                arch_name = architecture.architecture_name
                if not architecture.subnetworks:
                    self.logger.warning(f"No subnetworks found in architecture '{arch_name}'.")
                    continue
                if not architecture.subnetworks:
                    continue
                # Instatiate ANN Config instance
                ann_config_instances.append(
                    self._instantiate_and_format_class(
                        self.ontology.ANNConfiguration, arch_name
                    )
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

            # dataset_instance = self._instantiate_and_format_class(
            #     self.ontology.Dataset, "Dataset"
            # )
            # self._link_instances(
            #     dataset_pipe,
            #     dataset_instance,
            #     self.ontology.joinsDataSet,
            # )

            self.logger.info("Successfully processed training single.")
            self._process_dataset(training_step_instance)

        except Exception as e:
            self.logger.error(
                f"Error in _process_training_strategy: {e}", exc_info=True
            )
            raise

    # TODO: This is a temporary solution to add classes to the ontology.
    def __addclasses(self) -> None:
        """Adds new predefined classes to the ontology."""

        # Add hasWeightInitialization
        create_class_object_property(self.ontology, "hasWeightInitialization", self.ontology.TrainingSingle, self.ontology.WeightInitialization)
        ######

        # Add tasks
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

    def run(self, ann_path: List[str]) -> None:
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

                def grab_titles():
                    unique_titles = set()
                    import json

                    # Loop through each file and extract titles
                    for file in self.list_json_doc_paths:
                        with open(file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for entry in data:
                                if "metadata" in entry and "title" in entry["metadata"]:
                                    unique_titles.add(entry["metadata"]["title"])
                    return list(unique_titles)

                # Initialize the LLM engine for each json_document context in paper and/or code.
                for count, j in enumerate(self.list_json_doc_paths):
                    init_engine(self.ann_config_name, j)

                self.__addclasses()  # Add new general classes to ontology #TODO: better logic for doing this elsewhere

                # Instantiate the ANN Configuration class.
                ann_config_instance = self._instantiate_and_format_class(
                    self.ontology.ANNConfiguration, self.ann_config_name
                )
                create_class_data_property(
                    self.ontology, "hasTitle", self.ontology.ANNConfiguration, str, True
                )
                for title in grab_titles():
                    ann_config_instance.hasTitle = title
                create_class_data_property(
                    self.ontology,
                    "hasPaperPath",
                    self.ontology.ANNConfiguration,
                    str,
                    True,
                )
                for path in ann_path:
                    ann_config_instance.hasPaperPath = path

                # Process the network class and it's components.
                self._process_network(ann_config_instance)

                # Process TrainingStrategy and it's components.
                # self._process_training_strategy(ann_config_instance)

                # Log time taken to instantiate the ANN ontology instance.
                minutes, seconds = divmod(time.time() - start_time, 60)
                self.logger.info(
                    f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds."
                )

                self.logger.info(
                    f"Ontology instantiation completed for {self.ann_config_name}.\n\n\n"
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
