import hashlib
import os
import time
import json
import glob
from typing import Dict, Any, Union, List, Optional
import warnings
from builtins import TypeError
from rapidfuzz import process, fuzz
from pydantic import BaseModel
from owlready2 import (
    Ontology,
    ThingClass,
    ObjectPropertyClass,
    DataPropertyClass,
    Thing,
)
from utils.known_layer_types import check_actfunc, check_pooling, check_norm
from utils.constants import Constants as C
from utils.annetto_utils import load_annetto_ontology
from utils.util import get_sanitized_attr
from utils.llm_service import init_engine, query_llm
from utils.pydantic_models import *
from utils.logger_util import get_logger
from utils.owl_utils import (
    create_cls_instance,
    assign_object_property_relationship,
    create_subclass,
    get_all_subclasses,
    create_class_data_property,
    link_data_property_to_instance,
    entitiy_exists,
    assign_object_property_relationship,
)

global logger
logger = get_logger("instantiate_annetto")

class OntologyProcessor:
    """
    A class to instantiate an annett-o ontology by processing each main component separately and linking them together.
    """

    def __init__(
        self,
        ann_path: str,
        ann_config_name: str,
        ontology: Ontology,
        ontology_output_filepath: str = C.ONTOLOGY.TEST_ONTOLOGY_PATH,
        pt_network_names: List[str]=[],
    ) -> None:
        """
        Initialize the OntologyProcessor class.
        # Args:
            ann_path (str): The path to the ANN configuration directory.
            ann_config_name (str): The name of the ANN configuration.
            ontology (str): The ontology.
            ontology_output_filepath (str): The .owl path to save the ontology file.
        """
        self.logger = logger
        if not isinstance(ann_config_name, str):
            self.logger.error(
                "Expected a string for ANN Configuration name.", exc_info=True
            )
            raise TypeError(
                "Expected a string for ANN Configuration name."
            )
        if not isinstance(ontology, Ontology):
            self.logger.error(
                "Expected a Owlready2 Ontology type for ontology.", exc_info=True
            )
            raise TypeError(
                "Expected a Owlready2 Ontology type for ontology."
            )
        if not os.path.isdir(ann_path):
            self.logger.error(
                "Expected a directory path for ANN Configuration.", exc_info=True
            )
            raise TypeError(
                "Expected a directory path for ANN Configuration."
            )
        if not ontology_output_filepath.endswith(".owl"):
            self.logger.error(
                "Expected a string for output OWL path ending with .owl.", exc_info=True
            )
            raise TypeError(
                "Expected a string for output OWL path ending with .owl."
            )
        if not pt_network_names:
            self.logger.warning(
                "No PyTorch networks specified. Proceeding with an empty list."
            )
            pt_network_names = []

        if not isinstance(pt_network_names, list):
            self.logger.warning(
                "Expected a list for pt_networks. Proceeding with an empty list."
            )
            pt_network_names = []

        if not all(isinstance(network, str) for network in pt_network_names):
            self.logger.warning(
                "Expected a list of strings for pt_networks. Proceeding with an empty list."
            )
            pt_network_names = []


        self.ontology = ontology
        self.ann_path = ann_path
        self.ann_config_name = ann_config_name.lower().strip()
        self.output_owl_path = ontology_output_filepath

        self.llm_cache: Dict[str, Any] = {}
        self.ann_config_hash = self._generate_hash(self.ann_config_name)
        self.pt_network_names = pt_network_names

    def _generate_hash(self, str: str) -> str:
        """
        Generate a unique hash identifier based on the given string.
        """
        try:
            hash_object = hashlib.md5(str.encode())  # Generate a consistent hash
            return hash_object.hexdigest()[:8]
        except Exception as e:
            self.logger.error(
                f"Error generating hash for string '{str}': {e}", exc_info=True
            )
            return str

    def _format_instance_name(self, instance_name: str) -> str:
        return f"{self.ann_config_hash}_{instance_name.replace(' ', '-').lower()}"

    def _unformat_instance_name(self, instance_name: str) -> str:
        parts = instance_name.split("_", 1)
        if len(parts) == 2:
            return " ".join(
                word.capitalize() for word in parts[1].replace("-", " ").split()
            )
        return instance_name

    def _instantiate_and_format_class(
        self, cls: ThingClass, instance_name: str, source: Optional[str] = None
    ) -> Optional[Thing]:
        """
        Instantiate a given ontology class with the specified instance name.
        Uses the ANN configuration hash as a prefix for uniqueness.
        :param cls: The ontology class to instantiate.
        :param instance_name: The name of the instance to create.
        :param source: Optional source for the instance (i.e. 'default', 'code', 'llm').
        :return: The instantiated Thing object.
        """
        try:
            unique_name = self._format_instance_name(instance_name)
            instance = create_cls_instance(self.ontology, cls, unique_name)
            if not isinstance(instance, Thing):
                raise TypeError(f"Instance is not of type Thing: {cls} {type(instance)}")
            self.logger.info(f"Instantiated {cls.name} as {unique_name}.")
            if source:
                self._add_source_data_property(instance, source)
            return instance
        except Exception as e:
            self.logger.error(
                f"Error instantiating {cls} with name '{instance_name}'.",
                exc_info=True,
            )
            return None

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
        try:
            if not isinstance(instance_name, str):
                raise TypeError("Expected instance_name to be a string.", exc_info=True)
            if not all(isinstance(cls, (ThingClass, Thing)) for cls in classes):
                raise TypeError(
                    "Expected classes to be a list of ThingClass objects.", exc_info=True
                )
            if not all(isinstance(cls.name, str) for cls in classes):
                raise TypeError(
                    "Expected classes to have string names. ######", exc_info=True
                )
            if not isinstance(threshold, int):
                raise TypeError("Expected threshold to be an integer.", exc_info=True)
            if not classes:
                return None

            # Convert classes to a dictionary for lookup
            class_name_map = {cls.name.lower(): cls for cls in classes}

            match, score, _ = process.extractOne(
                instance_name.lower(), class_name_map.keys(), scorer=fuzz.ratio
            )
            # might need to reupper names later capitalized_string = string[0].upper() + string[1:]

            return class_name_map[match] if score >= threshold else None
        except Exception as e:
            self.logger.error(
                f"Error in fuzzy matching: {e}", exc_info=True
            )
            return None
        
    def _fuzzy_match_list(
        self, name: str, class_names: List[str], threshold: int = 80
    ) -> Optional[str]:
        """
        Perform fuzzy matching to find the best match for an instance in a list of strings.

        :param name: The item name.
        :param class_names: A list of string names to match with.
        :param threshold: The minimum score required for a match.
        :return: The best-matching original string or None if no good match is found.
        """
        try:
            if not all(isinstance(n, str) for n in class_names):
                raise TypeError("Expected class_names to be a list of strings.")
            if not isinstance(threshold, int):
                raise TypeError("Expected threshold to be an integer.")
            if not class_names:
                return None

            # Build a mapping from lowercased to original
            lower_to_original = {n.lower(): n for n in class_names}
            class_names_lower = list(lower_to_original.keys())

            match_lower, score, _ = process.extractOne(
                name.lower(), class_names_lower, scorer=fuzz.ratio
            )

            return lower_to_original[match_lower] if score >= threshold else None

        except Exception as e:
            self.logger.error(f"Error in fuzzy matching: {e}", exc_info=True)
            return None


    def _link_instances(
        self,
        parent_instance: Thing,
        child_instance: Thing,
        object_property: ObjectPropertyClass,
    ) -> None:
        """
        Link two instances via an object property.
        """
        try:
            assign_object_property_relationship(
                parent_instance, child_instance, object_property
            )
            self.logger.info(
                f"Linked instances '{parent_instance.name}' and '{child_instance.name}' via obj prop '{object_property}'."
            )
        except Exception as e:
            self.logger.error(
                f"Error linking instances '{parent_instance}' and '{child_instance}': {e}",
                exc_info=True,
            )

    def _link_data_property(
        self, instance: Thing, data_property: DataPropertyClass, value: Union[str, int, float, bool]
    ) -> None:
        """
        Link a data property to an instance.
        """
        link_data_property_to_instance(instance, data_property, value)
        self.logger.info(
            f"Linked '{instance.name}' with data property '{data_property.name}'."
        )

    def _add_source_data_property(self, instance: Thing, source: str) -> None:
        """
        Add a source data property to an instance.
        """
        try:
            link_data_property_to_instance(instance, self.ontology.sourceData, source)
            self.logger.info(
                f"Linked '{instance}' with source data property '{source}'."
            )
        except Exception as e:
            self.logger.error(
                f"Error linking source data property to instance '{instance}': {e}",
                exc_info=True,
            )

    def _add_definition_data_property(self, instance: Thing, definition: str) -> None:
        try:
            link_data_property_to_instance(instance, self.ontology.definition, definition)
            self.logger.info(
                f"Linked instance '{instance}' with definition data property."
            )
        except Exception as e:
            self.logger.error(
                f"Error linking definition data property to instance '{instance}': {e}",
                exc_info=True,
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
                token_budget=5000,
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
                obj_cls, f"{obj_type} Objective Function", "default"
            )
            self._link_instances(
                network_instance, obj_instance, self.ontology.hasObjective
            )
        else:
            obj_cls = self.ontology.MinObjectiveFunction
            obj_instance = self._instantiate_and_format_class(
                obj_cls, f"Min Objective Function", "llm"
            )
            self.logger.warning(
                f"No objective type specified for {network_name}. Defaulting to MinObjectiveFunction."
            )

        # Loss function handling
        if loss_name:
            known_losses = get_all_subclasses(self.ontology.LossFunction)
            best_loss_match = self._fuzzy_match_class(
                loss_name, known_losses, 90
            ) or create_subclass(self.ontology, loss_name, self.ontology.LossFunction)
            cost_instance = self._instantiate_and_format_class(
                self.ontology.CostFunction, "cost function", "default"
            )
            loss_instance = self._instantiate_and_format_class(
                best_loss_match, loss_name, "llm"
            )
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
            reg_instance = self._instantiate_and_format_class(
                best_reg_match, reg_name, "llm"
            )
            self._link_instances(
                cost_instance, reg_instance, self.ontology.hasRegularizer
            )
            if reg_def:
                self._add_definition_data_property(reg_instance, reg_def)

        self.logger.info(
            f"Processed objective functions for {network_name}: Loss: {loss_name}, Regularizer: {reg_name}, Objective: {obj_type}."
        )

    def _extract_network_data(self) -> list:
        """
        Extract name & type for each layer found within relevant parsed code

        :return List of dictionaries containing name & type per layer
        """
        network_json_path = glob.glob(f"data/{self.ann_config_name}/*.json")
        extracted_data: list = []

        for file in network_json_path:
            with open(file, "r") as f:
                try:
                    data: dict = json.load(f)

                    if "network" in data and isinstance(data["network"], list):
                        for layer in data["network"]:
                            if (
                                "name" in layer
                                and "type" in layer
                                and layer["type"] is not None
                            ):
                                extracted_data.append(
                                    {"name": layer["name"], "type": layer["type"]}
                                )
                except Exception as e:
                    self.logger.exception(
                        f"Error extracting JSON network data in {file} {e}",
                        exc_info=True,
                    )

        return extracted_data
    
    def _instantiate_temp_act_class(self , subclass_name: str="ActivationFunctionLayer") -> None:
        """
        Instantiate a subclass of ANNETT-O for a special case
        Currently used to instantiate 'ActivationFunctionLayer' as a subclass of layer

        :param subclass_name: Name of the new subclass
        :param entities: List of entities to be instantiated
        """
        subclass = create_subclass(self.ontology , subclass_name , self.ontology.Layer)
        actlayer_subclasses = get_all_subclasses(self.ontology.ActivationFunction)

        for entity in actlayer_subclasses:
            _ = create_subclass(
                self.ontology ,
                entity ,
                self.ontology.subclass
            )

    def _process_parsed_code(self , network_instance: Thing , module_names: List[str]=None , actfunc_flag: bool=True) -> None:
        """
        Process code that has been parsed into specific JSON structure to instantiate
        an ontology for the network associated with the code

        :param network_instance: the network instance
        :param actfunc_layer: (Default=True) Set to false to treat activation functions as layers
        :param module_name: (Optional) List of PyTorch module names to be instantiated
        :return None
        """
        try:
            json_files: list = []
            if module_names: # PyTorch only
                for module in module_names:
                    json_files.extend(glob.glob(f"{self.ann_path}/*{module}*.json"))
            else:
                json_files.extend(glob.glob(f"{self.ann_path}/**/*torch*.json" , recursive=True))
                json_files.extend(glob.glob(f"{self.ann_path}/**/*pb*.json" , recursive=True))
                json_files.extend(glob.glob(f"{self.ann_path}/**/*onnx*.json" , recursive=True))

            if not json_files:
                self.logger.error(f"No relevant JSON files (parsed code) found {module_names}")
                return

            # fetch ontology subclasses
            actlayer_subclasses: list = []
            if actfunc_flag:
                actlayer_subclasses = get_all_subclasses(self.ontology.ActivationLayer)
            else:
                actlayer_subclasses = get_all_subclasses(self.ontology.ActivationFunctionLayer)
            layer_subclasses: list = get_all_subclasses(self.ontology.Layer)
            pooling_subclasses: list = get_all_subclasses(self.ontology.PoolingLayer)
            norm_subclasses: list = get_all_subclasses(self.ontology.BatchNormLayer)

            for file in json_files:
                with open(file, "r") as f:
                    network_data: dict = json.load(f)

                nodes = network_data.get("graph", {}).get("node", [])
                if nodes is None:
                    warnings.warn("No parsed code available for given network")

                total_num_params: int = network_data.get("total_num_params")
                if total_num_params:
                    self._link_data_property(network_instance , self.ontology.total_network_params , total_num_params)
                    self.logger.info(f"Instantiated network {network_instance} with total number of parameters {total_num_params}")

                name_to_instance: dict = { # invoke fewer ontology calls
                    "instance": None,
                    "type": None
                }
                for layer in nodes:
                    layer_name = layer.get("name")

                    if not layer_name or layer_name == '':
                        continue

                    layer_type = layer.get("op_type")
                    layer_params = layer.get("num_params")
                    self.logger.info(
                        f"Instantiating layer {layer_name} in model {self.ann_config_name}"
                    )
                    self.logger.info(
                        f"{layer_name} INFO: type {layer_type} , params {layer_params}"
                    )

                    if not layer_type:
                        continue
                    
                    # fetch known layer types
                    known_actfunc: list = check_actfunc()
                    known_pooling: list = check_pooling()
                    known_norm: list = check_norm()

                    score: int = 70  # matching score (X / 100)
                    best_actfunc_match = self._fuzzy_match_list(
                        layer_type, known_actfunc, score
                    )
                    best_pooling_match = self._fuzzy_match_list(
                        layer_type, known_pooling, score
                    )
                    best_norm_match = self._fuzzy_match_list(
                        layer_type, known_norm, score
                    )

                    # check if special layer type, else treated as normal layer
                    if best_actfunc_match:
                        try:
                            actfunc_ontology = self._fuzzy_match_class(layer_type , actlayer_subclasses , score) # check the ontology for activation function
                            if not actfunc_ontology:
                                if actfunc_flag:
                                    actfunc_ontology = create_subclass(
                                        self.ontology , 
                                        layer_type , 
                                        self.ontology.ActivationFunction
                                    )
                                    self.logger.info(f"Activation layer {layer_name} subclass created in the ontology")
                                else:
                                    actfunc_ontology = create_subclass(
                                        self.ontology ,
                                        layer_type ,
                                        self.ontology.ActivationFunctionLayer
                                    )
                                    self.logger.info(f"Activation function layer {layer_name} subclass created in the ontology")

                            actlayer_subclasses.append(actfunc_ontology)
                            actfunc_instance = self._instantiate_and_format_class(actfunc_ontology , layer_name)
                            name_to_instance[layer_name] = {
                                "instance": actfunc_instance,
                                "layer_type": "activation"
                            }
                        except Exception as e:
                            self.logger.error(f"Error instantiating Activation layer {layer_name}: {e}" , exc_info=True)

                    elif best_pooling_match:
                        try:
                            pooling_ontology = self._fuzzy_match_class(layer_type , pooling_subclasses , score)
                            if not pooling_ontology:
                                pooling_ontology = create_subclass(
                                    self.ontology , 
                                    layer_type , 
                                    self.ontology.PoolingLayer
                                )
                                pooling_subclasses.append(pooling_ontology)
                                self.logger.info(f"Pooling layer {layer_name} subclass created in the ontology")
                            
                            pooling_instance = self._instantiate_and_format_class(pooling_ontology , layer_name)
                            self._link_instances(network_instance , pooling_instance , self.ontology.hasLayer)
                            name_to_instance[layer_name] = {
                                "instance": pooling_instance,
                                "layer_type": "pooling"
                            }
                        except Exception as e:
                            self.logger.error(f"Error instantiating Pooling layer {layer_name}: {e}" , exc_info=True)

                    elif best_norm_match:
                        try:
                            norm_ontology = self._fuzzy_match_class(layer_type , norm_subclasses , score)
                            if not norm_ontology:
                                norm_ontology = create_subclass(
                                    self.ontology , 
                                    layer_type , 
                                    self.ontology.BatchNormLayer
                                )
                                norm_subclasses.append(norm_ontology)
                                self.logger.info(f"Normalization layer {layer_name} subclass created in the ontology")

                            norm_instance = self._instantiate_and_format_class(norm_ontology , layer_name)
                            self._link_instances(network_instance , norm_instance , self.ontology.hasLayer)
                            name_to_instance[layer_name] = {
                                "instance": norm_instance,
                                "layer_type": "norm"
                            }
                        except Exception as e:
                            self.logger.error(f"Error instantiating Normalization layer {layer_name}: {e}" , exc_info=True)
                    
                    else: 
                        try:
                            best_layer_match = self._fuzzy_match_class(layer_type , layer_subclasses , score)
                            if not best_layer_match: # create subclass if layer type not found
                                best_layer_match = create_subclass(
                                    self.ontology , 
                                    layer_type , 
                                    self.ontology.Layer
                                )
                                layer_subclasses.append(best_layer_match)

                            layer_instance = self._instantiate_and_format_class(best_layer_match , layer_name)
                            self._link_instances(network_instance , layer_instance , self.ontology.hasLayer)

                            if layer_params:
                                self._link_data_property(layer_instance , self.ontology.layer_num_units , layer_params)
                            else:
                                self.logger.warning(f"Layer {layer_name} of type {layer_type} does not have a number of parameters")
                            name_to_instance[layer_name] = {
                                "instance": layer_instance,
                                "layer_type": layer_type
                            }
                        except Exception as e:
                            self.logger.error(f"Error instantiating layer {layer_name}: {e}")

                # second run for instantiating next, prev (skip and find next/prev for non-layers if actfunc_flag is set False)
                for layer in nodes:
                    layer_name = layer.get('name')

                    if not layer_name or layer_name == '':
                        continue

                    layer_type = layer.get('op_type')
                    prev_layers: list = layer.get('input' , [])
                    next_layers: list = layer.get('output' , [])

                    self.logger.info(f"I/O instantiation for layer {layer_name} in model {self.ann_config_name}")
                    self.logger.info(f"{layer_name}: input(s) {prev_layers} , output(s) {next_layers}")

                    layer_instance = name_to_instance.get(layer_name)
                    if not layer_instance:
                        self.logger.warning(f"Layer instance not found for {layer_name} , linkage unsuccessful")
                        continue
                    layer_instance_type = layer_instance.get("layer_type")
                    layer_instance = layer_instance.get("instance")
                    
                    for prev in prev_layers: # link prev_layer
                        prev_layer_name = name_to_instance.get(prev)
                        if not prev_layer_name:
                            self.logger.warning(f"Previous layer {prev_layer_name} could not be instantiated for layer {layer_name}")
                            continue
                        prev_layer_instance = prev_layer_name.get("instance")
                        prev_layer_type = prev_layer_name.get("layer_type")

                        if actfunc_flag: # skip linking activation layers
                            if prev_layer_type =="activation":
                                continue
                            if layer_instance_type == "activation":
                                continue
                        
                        self._link_instances(layer_instance , prev_layer_instance , self.ontology.previousLayer)
                        self.logger.info(f"Previous layer {prev} instantiated for layer {layer_name}")
                    
                    for next in next_layers: # link nextLayer & handle activation functions
                        next_layer_entry = name_to_instance.get(next)
                        if not next_layer_entry:
                            self.logger.warning(f"Next layer {next} could not be instantiated for layer {layer_name}")
                            continue
                        next_layer_instance = next_layer_entry.get("instance")
                        next_layer_type = next_layer_entry.get("layer_type")

                        if actfunc_flag and layer_instance_type == "activation": # skip activation function
                            self.logger.info(f"I/O instantiation skipped for activation function {layer_type} at {layer_name}")
                            continue

                        if next_layer_type == "activation": # handle activation layers
                            _ = create_subclass(
                                self.ontology,
                                layer_name, 
                                self.ontology.ActivationLayer
                            )

                            if actfunc_flag:
                                self._reassign_layer_linkage(layer_instance , next_layer_entry , next , next_layer_type , name_to_instance , nodes)
                                continue
                            else:
                                self._link_instances(layer_instance , next_layer_instance , self.ontology.hasActivationFunction) # parent hasActFunc
                                self._link_instances(network_instance , next_layer_instance , self.ontology.hasLayer)
                                #self._link_instances(network_instance , next_layer_instance , self.ontology.ActivationFunctionLayer) # child is now layer
                    
                        self._link_instances(layer_instance , next_layer_instance , self.ontology.nextLayer)
                        self.logger.info(f"Next layer {next} instantiated for {layer_name}")

                self.logger.info(f"Layers of {self.ann_config_name} processed & instantiated in ontology")
        except Exception as e:
            self.logger.error(f"Error processing parsed code for {self.ann_config_name}: {e}" , exc_info=True)

    def _reassign_layer_linkage(self, parent_layer: Thing, child_layer_entry: dict , child_layer_name: str, stored_type: str, name_to_instance: dict , node_data: dict) -> None:
        """
        Recursively reassign linkages between layer instances that are not layers

        :param parent_layer: The parent layer instance
        :param child_layer_entry: Dict of layer instance & type
        :param child_layer_name: Name identifying child layer
        :param stored_type: The stored type of the child layer (ex. activation)
        :param name_to_instance: Dictionary mapping layer names to their instance and type
        :param node_data: Network data from model's parsed JSON
        :return None or instance of the next layer
        """
        try:
            self.logger.info(f"Reassigning linkages for parent {parent_layer} to child {child_layer_name}")
            child_node = next((node for node in node_data if node.get("name") == child_layer_name) , None)

            if not child_node:
                self.logger.warning(f"Network data for layer {child_layer_name} not found.")
                return
            child_layer_instance = child_layer_entry.get("instance")

            # check if child is an activation function
            if stored_type == "activation":
                self._link_instances(parent_layer, child_layer_instance, self.ontology.hasActivationFunction)

                next_instance = None
                next_layer_names: list = child_node.get('output', [])

                for next_name in next_layer_names:
                    next_entry = name_to_instance.get(next_name)
                    if not next_entry:
                        self.logger.warning(f"No instance found for next layer {next_name}")
                        continue

                    next_instance = next_entry.get("instance")
                    next_type = next_entry.get("layer_type")

                    # recurse if there's another activation function
                    if next_type == "activation":
                        self._reassign_layer_linkage(parent_layer, next_entry , next_name, next_type, name_to_instance , node_data)
                    else:
                        self._link_instances(parent_layer, next_instance, self.ontology.nextLayer)
                        self._link_instances(next_instance, parent_layer, self.ontology.previousLayer)
                
                return next_instance
            else: # ez
                return
        except Exception as e:
            self.logger.error(f"Error re-assigning linkages for activation function {child_layer_name}: {e}" , exc_info=True)        
    
    def _llm_process_layers(self, network_instance: Thing) -> None:
        self.logger.error("Process layers with LLM not implemented yet.")
        return None

    def _process_layers(self, network_instance: Thing) -> None:
        """
        DEPRECATED
        Process the different layers (input, output, activation, noise, and modification) of it's network instance.

        :param network_instance: the network instance
        :return None
        """
        try:
            # fetch info from database
            onn = None
            onn.init_engine()
            models_list = onn.fetch_models()
            num_models = len(models_list)
            prev_model = None
            subclasses: List[ThingClass] = get_all_subclasses(self.ontology.Layer)

            #### NOTE
            # accounts for undetailed ontology wherein activation functions
            # are treated as layers (prevents duplication)
            subclasses.extend(get_all_subclasses(self.ontology.ActivationFunction))

            # fetch ann config name, find relevant model in database
            best_model_name = self._fuzzy_match_list(models_list)
            if not best_model_name:
                warnings.warn(f"Model name {best_model_name} not found in database")
                self._llm_process_layers(network_instance)  # for now...
                # throw to josue's script for llm instantiation?

            # fetch layer list of relevant model
            layer_list = onn.fetch_layers(best_model_name)

            for name in layer_list:
                layer_name, layer_type, model_id, model_name = name

                # Trying to be robust to weird db situations
                if layer_name is None:
                    continue
                if layer_name.lower() == self.ann_config_name.lower():
                    continue

                best_subclass_match = self._fuzzy_match_class(
                    layer_type, subclasses, 70
                )
                if (
                    not best_subclass_match
                ):  # create subclass if layer type not found in ontology
                    best_subclass_match = create_subclass(
                        self.ontology, layer_type, self.ontology.Layer
                    )
                    subclasses.append(
                        best_subclass_match
                    )  # track subclasses, ensure no duplicates

                # Debugging
                if model_id != prev_model:
                    self.logger.info(f"Processing model {model_id} / {num_models}")
                    self.logger.info(
                        f"Model name {model_name}, subclass match {best_subclass_match}"
                    )

                layer_instance = self._instantiate_and_format_class(
                    best_subclass_match, layer_name, "db"
                )
                self._link_instances(
                    network_instance, layer_instance, self.ontology.hasLayer
                )

            self.logger.info(f"All layers of {model_name} successfully processed")

        except Exception as e:
            self.logger.error(f"Error in _process_layers: {e}", exc_info=True)

    def _process_task_characterization(self, network_instance: Thing) -> None:
        network_name = self._unformat_instance_name(network_instance.name)

        try:
            if not isinstance(network_instance, Thing):
                self.logger.error(
                    "Expected an instance of Thing for Network in _process_task_characterization.",
                    exc_info=True,
                )
                raise ValueError(
                    "Expected an instance of Thing for Network in _process_task_characterization."
                )

            if not hasattr(self.ontology, "TaskCharacterization") or not isinstance(
                self.ontology.TaskCharacterization, ThingClass
            ):
                self.logger.error(
                    "The ontology must have a valid TaskCharacterization class of type ThingClass.",
                    exc_info=True,
                )
                return

            task = "Identify the machine learning *training task type* that this network is primarily designed to perform."

            query = f"Network: {network_name}"

            instructions = (
                "Return the response in JSON format with the key 'answer'.\n"
                # "The 'task_type' field should be a list of objects, each with a 'name' (e.g., 'Generation') "
                # "and an optional 'definition' (a succinct explanation of the term).\n"
                "The task type should reflect the machine learning training paradigm, not the specific application.\n"
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
            # task_type_def = get_sanitized_attr(response, "answer.task_type.definition")
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
                best_match_task_type, task_type_name, "llm"
            )

            self._link_instances(
                network_instance, task_type_instance, self.ontology.hasTaskType
            )

            # # Task definition handling
            # if task_type_def:
            #     self._add_definition_data_property(
            #         task_type_instance, task_type_def
            #     )

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
            ann_config_instance_name = self._unformat_instance_name(
                ann_config_instance.name
            )

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
- Should be described with the word 'network' in its name (e.g., 'attention network').\n
"""

            instructions = (
                "Return the response in JSON format with the key 'answer'.\n"
                "If a subnetwork is not named but has a described purpose, use a short functional label (e.g., “temporal integration module”).\n"
            )
            query = f""
            extra_instructions = (
                "Avoid listing low-level components like layers (e.g., convolution, fully-connected, pooling, activation) as separate subnetworks.\n"
                "If the architecture describes a single unified model composed of standard layers, return only one subnetwork with a high-level functional name (e.g., 'convolutional network' or 'feedforward network')."
            )

            prompt = self.build_prompt(
                task, query, instructions, examples, extra_instructions
            )
            response = self._query_llm(prompt, NetworkResponse)
            if not response or not response.answer.architecture:
                self.logger.warning(
                    f"No architectures found in ANN '{ann_config_instance_name}'."
                )
                return

            details = response.answer.architecture
            if not details.subnetworks:
                self.logger.warning(
                    f"No subnetworks found for ANN '{ann_config_instance_name}'."
                )
                return
            
            # pt networks list
            unused_pt_networks:List[str] = self.pt_network_names
            # List for keeping track of processed networks from the LLM
            unused_subnetwork_names: List[str] = []

            for subnetwork in details.subnetworks:
                sub_name = subnetwork.name
                is_independent = subnetwork.is_independent
                self.logger.info(
                    f"ANN: {ann_config_instance_name}, Subnetwork: {sub_name} is independent: {is_independent}"
                )
                # Skip if not an independent network
                # This account for the LLM including layers as subnetworks
                # check if any word in sub_name is 'layer' or 'layers'
                if any(
                    word in sub_name.lower()
                    for word in ["layer", "layers", "pooling", "activation", "pool"]
                ):
                    self.logger.warning(
                        f"Subnetwork '{sub_name}' is mentioned as a layer. Skipping."
                    )
                    continue
                if not is_independent:
                    self.logger.warning(
                        f"Subnetwork '{sub_name}' is not independent. Skipping."
                    )
                    continue
                unused_subnetwork_names.append(sub_name)
            if not unused_subnetwork_names:
                self.logger.error(
                    f"No subnetwork names found for ANN '{ann_config_instance_name}'."
                )
                return
                        
            network_matches: dict = {} # for keeping track of matched networks with pt networks
        
            for subname in unused_subnetwork_names:
                match = self._fuzzy_match_list(subname, unused_pt_networks, threshold=50)
                if match:
                    unused_pt_networks.remove(match)
                    unused_subnetwork_names.remove(subname)
                    network_matches[subname] = match
            self.logger.info(f"Network matches in ANN '{ann_config_instance_name}': '{network_matches}'")

            if not hasattr(self.ontology, "ParentNetwork"):
                logger.error(
                    "Invalid or missing ParentNetwork class in the ontology.",
                    exc_info=True,
                )
                raise AttributeError(
                    "The ontology must have a valid 'ParentNetwork' class of type ThingClass."
                )
            
            networks_to_process: List[Thing] = []

            if unused_subnetwork_names and unused_pt_networks:
                print("case 1")
                # parse layers into Parent Network
                parentnetwork_instance = self._instantiate_and_format_class(
                    self.ontology.ParentNetwork, ann_config_instance_name + " network", "default"
                )
                self._link_instances(
                    ann_config_instance,
                    parentnetwork_instance,
                    self.ontology.hasNetwork,
                )
                self._process_parsed_code(parentnetwork_instance, unused_pt_networks)
                # process remaining subnetwork components
                networks_to_process = unused_subnetwork_names
            elif not unused_subnetwork_names and unused_pt_networks and not network_matches:
                print("case 2")
                # parse layers into Parent Network
                parentnetwork_instance = self._instantiate_and_format_class(
                    self.ontology.ParentNetwork, ann_config_instance_name + " network", "default"
                )
                self._link_instances(
                    ann_config_instance,
                    parentnetwork_instance,
                    self.ontology.hasNetwork,
                )
                self._process_parsed_code(parentnetwork_instance, unused_pt_networks)
                # process parenet network components
                networks_to_process = [parentnetwork_instance]
            elif not unused_subnetwork_names and unused_pt_networks and network_matches:
                print("case 3")
                # parse layers into Parent Network
                parentnetwork_instance = self._instantiate_and_format_class(
                    self.ontology.ParentNetwork, ann_config_instance_name + " network", "default"
                )
                self._link_instances(
                    ann_config_instance,
                    parentnetwork_instance,
                    self.ontology.hasNetwork,
                )
                self._process_parsed_code(parentnetwork_instance, unused_pt_networks)
            elif unused_subnetwork_names and not unused_pt_networks:
                print("case 4")
                # process remaining subnetwork components
                networks_to_process = unused_subnetwork_names     

            # add mapped networks to networks to process
            networks_to_process.extend(network_matches.keys())
            print(f"Networks to process: {networks_to_process}")     
            self.logger.info(f"Networks to process in ANN '{ann_config_instance_name}': '{networks_to_process}'")  

            if not networks_to_process:
                self.logger.error(
                    f"No networks to process for ANN '{ann_config_instance_name}'."
                )
                return

            for subnetwork in networks_to_process:
                # Instantiate subnetwork instance
                network_instance = self._instantiate_and_format_class(
                    self.ontology.Network, subnetwork, "llm"
                )
                self._link_instances(
                    ann_config_instance,
                    network_instance,
                    self.ontology.hasNetwork,
                )

                if subnetwork in network_matches.keys():
                    match = network_matches[subnetwork]
                    self._process_parsed_code(network_instance, [match])
                self._process_objective_functions(network_instance)
                self._process_task_characterization(network_instance)
                self.logger.info(
                    f"Successfully processed network instance: '{network_instance.name}'"
                )
        except Exception as e:
            self.logger.error(
                f"Error processing the '{ann_config_instance}' networks: {e}",
                exc_info=True,
            )
            raise e

    def _process_dataset(self, training_step_instance: Thing) -> None:
        try:
            if not training_step_instance:
                self.logger.error(
                    "No TrainingStepInstance instance provided when processing dataset.",
                    exc_info=True,
                )
                raise ValueError(
                    "No TrainingStepInstance instance provided when processing dataset."
                )

            task = "Extract and describe all datasets used in the paper."

            extra_instructions = (
                "Each dataset must be described as an object with the following keys:\n\n"
                "Required:\n"
                "- data_description\n"
                "- data_type\n\n"
                "Optional:\n"
                "- dataset_name\n"
                "- data_doi\n"
                "- data_sample_dimensionality\n"
                "- data_samples\n"
                "- number_of_classes\n\n"
                "For 'data_type', suggested values include 'Image', 'MultiDimensionalCube', 'Text', and 'Video'. "
                "However, you may use other appropriate types if relevant."
            )

            extra_instructions = ""

            instructions = (
                "Return the response as a JSON object with a single key 'answer', "
                "which is a list of dataset objects matching the following format. "
                "Only include fields if they are explicitly stated or reasonably inferable. "
                "Unknown optional fields can be omitted or set to null."
            )

            examples = (
                "Example:\n"
                """{
                "answer": [
                    {
                        "data_description": "The MNIST handwritten digit dataset containing 60,000 training and 10,000 test examples.",
                        "dataset_name": "MNIST",
                        "data_doi": "10.1109/CVPR.2017.90",
                        "data_sample_dimensionality": "28x28",
                        "data_samples": 70000,
                        "number_of_classes": 10,
                        "data_type": "Image"
                    },
                    {
                        "data_description": "A synthetic dataset of generated sentences for augmenting training data.",
                        "data_type": "Text"
                    }
                ]
            }"""
            )

            prompt = self.build_prompt(
                task, "", instructions, examples, extra_instructions=extra_instructions
            )

            dataset_response = self._query_llm(
                prompt, pydantic_type_schema=MultiDatasetResponse
            )
            if not dataset_response:
                self.logger.warning("No dataset response received.")
                return

            counter = 0

            for dataset in dataset_response.answer:
                counter += 1
                self.logger.info("Starting dataset processing.")

                # Link Training Single to new DataPipe
                dataset_pipe_instance = self._instantiate_and_format_class(
                    self.ontology.DatasetPipe, f"Dataset Pipe {counter}", "default"
                )
                self._link_instances(
                    training_step_instance,
                    dataset_pipe_instance,
                    self.ontology.trainingSingleHasIOPipe,
                )

                # Check if dataset name exists, otherwise give a placeholder name
                if dataset.dataset_name:
                    dataset_instance = self._instantiate_and_format_class(
                        self.ontology.Dataset, dataset.dataset_name, "llm"
                    )
                else:
                    dataset_instance = self._instantiate_and_format_class(
                        self.ontology.Dataset, f"Unknown Dataset {counter}", "default"
                    )

                self._link_instances(
                    dataset_pipe_instance, dataset_instance, self.ontology.joinsDataSet
                )

                # Set Data Properties
                self._link_data_property(
                    dataset_instance,
                    self.ontology.data_description,
                    dataset.data_description,
                )
                if dataset.data_doi:
                    self._link_data_property(
                        dataset_instance,
                        self.ontology.data_doi,
                        dataset.data_doi,
                    )

                if dataset.data_sample_dimensionality:
                    self._link_data_property(
                        dataset_instance,
                        self.ontology.data_sample_dimensionality,
                        dataset.data_sample_dimensionality,
                    )
                if dataset.data_samples:
                    self._link_data_property(
                        dataset_instance,
                        self.ontology.data_samples,
                        dataset.data_samples,
                    )

                # Instantiate Label set for labeled data information
                label_set_instance = self._instantiate_and_format_class(
                    self.ontology.Labelset, "Label Set", "default"
                )
                self._link_instances(
                    dataset_instance,
                    label_set_instance,
                    self.ontology.hasLabels,
                )
                # Set number of classes
                if dataset.number_of_classes:
                    self._link_data_property(
                        label_set_instance,
                        self.ontology.labels_count,
                        dataset.number_of_classes,
                    )

                # Handle dataType
                data_type_subclass = dataset.data_type
                best_match = self._fuzzy_match_class(
                    data_type_subclass, get_all_subclasses(self.ontology.DataType)
                )
                if best_match:
                    self.logger.info(f"Matched DataType subclass: {best_match}")
                    data_type_instance = self._instantiate_and_format_class(
                        best_match, "Data Type", "default"
                    )
                    self._link_instances(
                        dataset_instance, data_type_instance, self.ontology.hasDataType
                    )
                else:
                    self.logger.info(
                        f"Unrecognized DataType subclass: {data_type_subclass}"
                    )

            self.logger.info("Completed dataset processing.")

        except Exception as e:
            self.logger.error(f"Error while processing dataset: {e}", exc_info=True)
            raise

    def _process_training_strategy(self, ann_config_instance: Thing) -> None:
        if not ann_config_instance:
            self.logger.error("No ANN Configuration instance provided.")
            raise ValueError("No ANN Configuration instance in the ontology.")

        if not hasattr(self.ontology, "TrainingStrategy"):
            self.logger.error("TrainingStrategy class not found in the ontology.")
            raise AttributeError("TrainingStrategy class not found in the ontology.")

        try:
            strategy_instance = self._instantiate_and_format_class(
                self.ontology.TrainingStrategy, "Training Strategy", "default"
            )
            self._link_instances(
                ann_config_instance,
                strategy_instance,
                self.ontology.hasPrimaryTrainingSession,
            )

            session_instance = self._instantiate_and_format_class(
                self.ontology.TrainingSession, "Training Session", "default"
            )
            self._link_instances(
                strategy_instance,
                session_instance,
                self.ontology.hasPrimaryTrainingSession,
            )
            self.logger.info("Successfully processed training strategy.")

        except Exception as e:
            self.logger.error(f"Error in _process_training_single: {e}", exc_info=True)
            raise

        self._process_training_single(ann_config_instance, session_instance)

    def _process_training_single(
        self, ann_config_instance: Thing, session_instance: Thing
    ) -> None:
        if not hasattr(self.ontology, "TrainingSingle"):
            self.logger.error("TrainingSingle class not found in the ontology.")
            raise AttributeError("TrainingSingle class not found in the ontology.")

        try:
            training_step_instance = self._instantiate_and_format_class(
                self.ontology.TrainingSingle, "Training Single", "default"
            )
            self._link_instances(
                session_instance,
                training_step_instance,
                self.ontology.hasPrimaryTrainingStep,
            )

            task = "Extract the training details used in the network-specific training step."
            instructions = (
                "Return a JSON object with key 'answer', containing:\n"
                "- batch_size: int\n"
                "- learning_rate_decay: float\n"
                "- number_of_epochs: int\n"
                "- learning_rate_decay_epochs (optional): int or null"
            )
            query = ""
            examples = (
                "{\n"
                '  "answer": {\n'
                '    "batch_size": 32,\n'
                '    "learning_rate_decay": 0.01,\n'
                '    "number_of_epochs": 10,\n'
                '    "learning_rate_decay_epochs": 5\n'
                "  }\n"
                "}\n"
                "If the learning_rate_decay_epochs value is not available, you may return None.\n\n"
            )
            extra_instructions = "If the learning_rate_decay_epochs value is not available, you may return None."
            prompt = self.build_prompt(
                task, query, instructions, examples, extra_instructions
            )

            response = self._query_llm(
                prompt,
                TrainingSingleResponse,
            )

            if not response:
                self.logger.warning("No training detail response received.")
                return
            details = response.answer
            # Set data properties
            if details.learning_rate_decay:
                self._link_data_property(
                    training_step_instance,
                    self.ontology.learning_rate_decay,
                    details.learning_rate_decay,
                )
                self._link_data_property(
                    training_step_instance,
                    self.ontology.learning_rate_decay,
                    details.learning_rate_decay,
                )

            if details.learning_rate_decay_epochs:
                self._link_data_property(
                    training_step_instance,
                    self.ontology.learning_rate_decay_epochs,
                    details.learning_rate_decay_epochs,
                )
            if details.number_of_epochs:
                self._link_data_property(
                    training_step_instance,
                    self.ontology.number_of_epochs,
                    details.number_of_epochs,
                )
            if details.batch_size:
                self._link_data_property(
                    training_step_instance, self.ontology.batch_size, details.batch_size
                )

            self.logger.info("Successfully processed training single.")
            self._process_dataset(training_step_instance)

        except Exception as e:
            self.logger.error(
                f"Error in _process_training_strategy: {e}", exc_info=True
            )
            raise

    def save_ontology(self) -> None:
        try:
            self.ontology.save(file=self.output_owl_path, format="rdfxml")
            self.logger.info(f"Ontology saved to {self.output_owl_path}.")
        except Exception as e:
            self.logger.error("Failed to save ontology.", exc_info=True)

    def run(self, ann_path: List[str]) -> None:
        try:
            with self.ontology:
                start_time = time.time()

                def _add_ann_metadata(
                    ann_config_instance: Thing,
                    titles: List[str],
                    paper_paths: List[str],
                ) -> None:
                    if not entitiy_exists(self.ontology, "hasTitle"):
                        create_class_data_property(
                            self.ontology,
                            "hasTitle",
                            self.ontology.ANNConfiguration,
                            str,
                            True,
                        )
                        for title in titles:
                            ann_config_instance.hasTitle = title

                    if not entitiy_exists(self.ontology, "hasPaperPath"):
                        create_class_data_property(
                            self.ontology,
                            "hasPaperPath",
                            self.ontology.ANNConfiguration,
                            str,
                            True,
                        )
                        for path in paper_paths:
                            ann_config_instance.hasPaperPath = path

                def _extract_titles_from_docs(json_doc_paths: List[str]) -> List[str]:
                    unique_titles = set()
                    for file in json_doc_paths:
                        with open(file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for entry in data:
                                title = entry.get("metadata", {}).get("title")
                                if title:
                                    unique_titles.add(title)
                    return list(unique_titles)

                json_pdf_paths = glob.glob(
                    f"{self.ann_path}/*pdf*.json"
                )  # Grabs all pdf doc json's
                json_code_paths = glob.glob(
                    f"{self.ann_path}/*code_doc*.json"
                )
                json_paths = json_pdf_paths + json_code_paths
                if not json_paths:
                    raise FileNotFoundError(
                        f"No JSON doc files found in {self.ann_path}."
                    )
                if not all(item.endswith(".json") for item in json_paths):
                    raise ValueError(
                        "All items in list_json_doc_paths must end with .json"
                    )

                if not hasattr(self.ontology, "ANNConfiguration"):
                    raise AttributeError(
                        "Class 'ANNConfiguration' not found in ontology."
                    )

                # Add docs to llm engine for RAG
                for j in json_paths:
                    init_engine(self.ann_config_name, j)

                # Instantiate ANN Configuration instance using parameter passed name
                ann_config_instance = self._instantiate_and_format_class(
                    self.ontology.ANNConfiguration, self.ann_config_name, "user"
                )

                # Extract metadata and attach to ANNConfiguration instance
                titles = _extract_titles_from_docs(json_paths)
                _add_ann_metadata(ann_config_instance, titles, ann_path)

                # Process the ANN Configuration instance
                self._process_network(ann_config_instance)
                self._process_training_strategy(ann_config_instance)

                minutes, seconds = divmod(time.time() - start_time, 60)
                self.logger.info(f"Elapsed time: {int(minutes)}m {seconds:.2f}s.")
                self.logger.info(
                    f"Ontology instantiation completed for {self.ann_config_name}.\n"
                )
                self.save_ontology()

        except Exception as e:
            self.logger.error(
                f"Error during ontology instantiation: {e}", exc_info=True
            )
            raise

def instantiate_annetto(
    ann_name: str, ann_path: str, ontology: Ontology, ontology_output_filepath: str, pt_network_names: List[str] = []
):
    """
    Instantiates an ANN ontology from the provided ANN Configuration filepath.
    Papers and Code must be extracted to the proper JSON format beforehand.

    :param: ann_name: The name of the ANN Configuration.
    :param: ann_path: The path to the ANN Configuration JSON files.
    :param: ontology: The ontology to instantiate.
    :param: ontology_output_filepath: The .owl file path of the ontology.
    """
    if not os.path.isdir(ann_path):
        raise NotADirectoryError(f"Path {ann_path} is not a directory.")
    if not isinstance(ontology, Ontology):
        raise TypeError("Ontology must be an instance of the Ontology class.")
    if not ontology_output_filepath.endswith(".owl"):
        raise ValueError("Ontology output file must have a .owl extension.")
    if not hasattr(ontology, "ANNConfiguration"):
        raise AttributeError("Error: Class 'ANNConfiguration' not found in ontology.")
    if not pt_network_names:
        warnings.warn(
            "No PyTorch networks were passed to instantiate_annetto."
        )
    if not isinstance(pt_network_names, list):
        warnings.warn(
            "PyTorch networks should be passed as a list."
        )
    if not all(isinstance(network, str) for network in pt_network_names):
        warnings.warn(
            "All PyTorch networks should be strings."
        )
    instantiator = OntologyProcessor(
        ann_path,
        ann_name,
        ontology=ontology,
        ontology_output_filepath=ontology_output_filepath,
        pt_network_names=pt_network_names,
    )
    instantiator.run(ann_path)
    print(
        f"Ontology instantiation completed for {ann_name} and saved to {ontology_output_filepath}."
    )


# For standalone testing
if __name__ == "__main__":
    time_start = time.time()

    ###################################################################
    def mass_instantiation(root_path: str, ontology: Ontology) -> None:
        """
        Mass instantiation of ANN configurations from a given root path.
        Expects a directory structure where each subdirectory contains ANN configuration files.
        """

        def get_subdirectories(path):
            return [f.name for f in os.scandir(path) if f.is_dir()]

        subdirectories = get_subdirectories(root_path)

        model_names_from_directory = [
            os.path.basename(subdirectory) for subdirectory in subdirectories
        ]
        from utils.owl_utils import save_ontology

        save_ontology(ontology, C.ONTOLOGY.TEST_ONTOLOGY_PATH)
        curr_ontology = load_annetto_ontology(return_onto_from_release="test")

        for model_name in model_names_from_directory:
            print(
                f"Model name: {model_name}, Path: {os.path.join(root_path, model_name)}"
            )
            try:
                instantiate_annetto(
                    model_name,
                    os.path.join(root_path, model_name),
                    curr_ontology,
                    C.ONTOLOGY.TEST_ONTOLOGY_PATH,
                )

            except Exception as e:
                print(f"Error instantiating {model_name}: {e}")

    def single_instantiation(model_name: str, ontology: Ontology) -> None:
        """
        Single instantiation of an ANN configuration.
        """
        ann_path = os.path.join("data", model_name)
        instantiate_annetto(
            model_name,
            ann_path,
            ontology,
            C.ONTOLOGY.TEST_ONTOLOGY_PATH,
        )

    ###################################################################

    ontology = load_annetto_ontology(return_onto_from_release="base")
    from src.ontology_population.initialize_annetto import initialize_annetto

    initialize_annetto(ontology, logger)
    """
    Comment out the following blocks if you want to add new classes to the ontology at mass or little at a time.
    """
    ### For single testing ###
    for model_name in [
        "alexnet",
        # "resnet",
        # "vgg16",
        # "gan",  # Assume we can model name from user or something
    ]:
        single_instantiation(model_name, ontology)

    ### For mass instantiation from a root dir ###
    # mass_instantiation("data/more_papers", ontology)

    time_end = time.time()
    minutes, seconds = divmod(time_end - time_start, 60)
    print(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds.")
