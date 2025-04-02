import logging
import os
import hashlib
from datetime import datetime
import time
from typing import Dict, Any, Union, List, Optional
import warnings
from builtins import TypeError

from owlready2 import Ontology, ThingClass, Thing, ObjectProperty
from rapidfuzz import process, fuzz
from pydantic import BaseModel

from utils.owl_utils import (
    create_cls_instance,
    assign_object_property_relationship,
    create_subclass,
    get_all_subclasses,
    create_class_data_property
)
from utils.constants import Constants as C

from utils.annetto_utils import int_to_ordinal, load_annetto_ontology

# from utils.onnx_db import OnnxAddition
from src.instantiate_annetto.prompt_builder import PromptBuilder

from utils.llm_service_simple_test import init_engine, query_llm
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
        list_json_doc_paths: List[str],
        ann_config_name: str,
        ontology: Ontology = load_annetto_ontology("base"),
        output_owl_path: str = C.ONTOLOGY.TEST_ONTOLOGY_PATH,
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
            self.logger.error("Expected a string for ANN Configuration name.", exc_info=True)
            raise TypeError("Expected a string for ANN Configuration name.", exc_info=True)
        if not isinstance(ontology, Ontology):
            self.logger.error("Expected a Owlready2 Ontology type for ontology.", exc_info=True)
            raise TypeError("Expected a Owlready2 Ontology type for ontology.", exc_info=True)
        if not isinstance(list_json_doc_paths, list) and all(
            isinstance(path, str) for path in list_json_doc_paths
        ):
            self.logger.error("Expected a list of strings for JSON doc paths.", exc_info=True)
            raise TypeError("Expected a list of strings for JSON doc paths.", exc_info=True)
        if not isinstance(output_owl_path, str):
            self.logger.error("Expected a string for output OWL path.", exc_info=True)
            raise TypeError("Expected a string for output OWL path.", exc_info=True)

        self.ontology = ontology
        self.list_json_doc_paths = list_json_doc_paths
        self.ann_config_name = ann_config_name.lower().strip()
        self.output_owl_path = output_owl_path

        self.llm_cache: Dict[str, Any] = {}
        self.logger = logger
        self.ann_config_hash = self._generate_hash(self.ann_config_name)
        self.prompt_builder = PromptBuilder()
        self._setup_prompt_examples()

    def _setup_prompt_examples(self):
        """Populate PromptBuilder with common examples."""
        # Objective Functions
        self.prompt_builder.add_example(
            "objective_function",
            'Network: Discriminator\n{"answer": {"loss": "Power-Outlet Loss", "regularizer": "Electric Regularization", "objective": "minimize"}}',
        )
        self.prompt_builder.add_example(
            "objective_function",
            'Network: Generator\n{"answer": {"loss": "Power-Outlet Loss", "regularizer": null, "objective": "maximize"}}',
        )

        # Input Layer
        self.prompt_builder.add_example(
            "input_layer", 'Network: SVM\n{"answer": "128x128x3"}'
        )
        self.prompt_builder.add_example(
            "input_layer", 'Network: Generator\n{"answer": 100}'
        )

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
        try:
            unique_instance_name = self._format_instance_name(instance_name)
            instance = create_cls_instance(self.ontology, cls, unique_instance_name)
            if not isinstance(instance, Thing):
                raise TypeError(f"{instance} is not a Thing: {type(instance)}")

            self.logger.info(
                f"Instantiated {cls.name} with name: {self._unformat_instance_name(unique_instance_name)}."
            )
            return instance
        except Exception as e:
            self.logger.error(
                f"Error instantiating {cls.name} with name {instance_name}: {e}", exc_info=True
            )

    def _format_instance_name(self, instance_name: str) -> str:
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

    def _unformat_instance_name(self, instance_name: str) -> str:
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
            raise TypeError("Expected classes to be a list of ThingClass objects.", exc_info=True)
        if not all(isinstance(cls.name, str) for cls in classes):
            raise TypeError("Expected classes to have string names. ######", exc_info=True)
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.", exc_info=True)

        # Convert classes to a dictionary for lookup
        class_name_map = {cls.name: cls for cls in classes}

        match, score, _ = process.extractOne(
            instance_name, class_name_map.keys(), scorer=fuzz.ratio
        )

        return class_name_map[match] if score >= threshold else None

    def _fuzzy_match_list(
        self, class_names: List[str], threshold: int = 80
    ) -> Optional[str]:
        """
        Perform fuzzy matching to find the best match for an instance in a list of strings.

        :param instance_name: The instance name.
        :param class_names: A list of string names to match with.
        :param threshold: The minimum score required for a match.
        :return: The best-matching string or None if no good match is found.
        """
        if not all(isinstance(name, str) for name in class_names):
            raise TypeError("Expected class_names to be a list of strings.", exc_info=True)
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.", exc_info=True)

        match, score, _ = process.extractOne(
            self.ann_config_name, class_names, scorer=fuzz.ratio
        )

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
            f"Linked {self._unformat_instance_name(parent_instance.name)} and {self._unformat_instance_name(child_instance.name)} via {object_property.name}."
        )
    def build_prompt(self, task: str, query: str, instructions: str, examples: str, extra_instructions: str = "") -> str:
        return f"{task}\n{instructions}\n{examples}\n{extra_instructions}\nNow, for the following:\n{query}\nAnswer: "

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
            )

            self.logger.info(f"LLM query: {prompt}")
            self.logger.info(f"LLM query response: {response}")
            self.llm_cache[prompt] = response

            return response
        except Exception as e:
            self.logger.error(f"LLM query error: {e}", exc_info=True)
            return ""
        
    def _process_objective_functions(self, network_instance: Thing) -> None:
        network_name = self._unformat_instance_name(network_instance.name)

        # examples = (
        #     "Examples:\n"
        #     "Network: Discriminator\n"
        #     '''{"answer": {"loss": "Power-Outlet Loss", "regularizer": "Electric Regularization", "objective": "minimize"}}\n\n'''

        #     "Network: Generator\n"
        #     '''{"answer": {"loss": "Power-Outlet Loss", "regularizer": null, "objective": "maximize"}}\n'''
        # )

        # Define examples using defintions
        examples = (
            "Examples:\n"
            "Network: Discriminator\n"
            '''{"answer": {
                "loss": {
                    "name": "Power-Outlet Loss",
                    "definition": "Measures energy imbalance between predicted and real outputs."
                },
                "regularizer": {
                    "name": "Electric Regularization",
                    "definition": "Penalizes current surges to stabilize the model."
                },
                "objective": "minimize"
            }}\n\n'''

            "Network: Generator\n"
            '''{"answer": {
                "loss": {
                    "name": "Power-Outlet Loss",
                    "definition": "Measures energy imbalance between predicted and real outputs."
                },
                "regularizer": null,
                "objective": "maximize"
            }}\n'''
        )


        # Define the task directly
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

        # prompt = (
        #     f"Extract the loss function, regularizer, and objective type for a network.\n"
        #     f"{examples}\n"
        #     f"Now, for the following network:\n"
        #     f"Network: {network_name}\n"
        #     "Objective type must be 'minimize' or 'maximize'. Regularizer can be null if not specified."
        # )

        # # Define JSON format instructions manually
        # json_format = (
        #     "Return the response in JSON format with the key 'answer'. The value should be an object with:\n"
        #     "- 'loss': The loss function name (string)\n"
        #     "- 'regularizer': The regularizer function name (string or null)\n"
        #     "- 'objective': The objective type ('minimize' or 'maximize')\n"
        #     "Example:\n"
        #     '{"answer": {"loss": "MSE", "regularizer": "L1", "objective": "maximize"}}'
        # )

        prompt = self.build_prompt(task, query, instructions, examples, extra_instructions)

        # Query LLM
        response = self._query_llm(prompt, ObjectiveFunctionResponse)
        if not response:
            self.logger.warning(f"No response for objective functions in network {network_name}.")
            return

        # Rest of the method remains unchanged
        # loss_name = str(response.answer.cost_function.lossFunction)
        # if response.answer.cost_function.regularFunction:
        #     reg_name = str(response.answer.cost_function.regularFunction)
        # obj_type = str(response.answer.objectiveFunction)
        loss_name = str(response.loss.name)
        loss_def= str(response.loss.definition)

        reg_name = None
        if response.regularizer:
            reg_name = str(response.regularFunction.name)
            reg_def = str(response.regularFunction.defintion)

        obj_type = str(response.objective)



        # Instantiate and link (rest of your logic remains similar)
        obj_cls = (
            self.ontology.MinObjectiveFunction
            if obj_type.lower().strip() == "minimize"
            else self.ontology.MaxObjectiveFunction
        )
        obj_instance = self._instantiate_and_format_class(
            obj_cls, f"{obj_type} Objective Function"
        )
        self._link_instances(network_instance, obj_instance, self.ontology.hasObjective)

        # Loss function handling
        known_losses = get_all_subclasses(self.ontology.LossFunction)
        best_loss_match = self._fuzzy_match_class(
            loss_name, known_losses, 90
        ) or create_subclass(self.ontology, loss_name, self.ontology.LossFunction)
        cost_instance = self._instantiate_and_format_class(
            self.ontology.CostFunction, "cost function"
        )
        loss_instance = self._instantiate_and_format_class(best_loss_match, loss_name)
        self._link_instances(obj_instance, cost_instance, self.ontology.hasCost)
        self._link_instances(cost_instance, loss_instance, self.ontology.hasLoss)

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

        self.logger.info(
            f"Processed objective functions for {network_name}: Loss: {loss_name}, Regularizer: {reg_name}, Objective: {obj_type}."
        )

    def _old_process_objective_functions(self, network_instance: Thing) -> None:
        if not isinstance(network_instance, Thing):
            self.logger.error(
                "Network instance not provided in process_objective_functions."
            )
            raise TypeError("Expected an instance of 'Thing' for Network.", exc_info=True)

        network_name = self._unformat_instance_name(network_instance.name)

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            task=f"Extract the loss function, regularizer, and objective type for a network.",
            query=f"Network: {network_name}",
            category="objective_function",
            extra_instructions=(
                "Objective type must be 'minimize' or 'maximize'. "
                "Regularizer can be null if not specified."
            ),
        )

        # Define JSON format with Pydantic schema in mind
        fields = {
            "loss": "The loss function name (string)",
            "regularizer": "The regularizer function name (string or null)",
            "objective": "The objective type ('minimize' or 'maximize')",
        }
        example_output_1 = {
            "answer": {
                "loss": "MSE",
                "regularizer": "L1",
                "objective": "maximize",
            }
        }
        json_format = self.prompt_builder.build_json_format_instructions(
            fields, example_output_1
        )

        # Query LLM
        response = self._query_llm("", prompt, json_format, ObjectiveFunctionResponse)
        if not response:
            self.logger.warning(
                f"No response for objective functions in network {network_name}."
            )
            return

        # Process response
        loss_name = str(response.answer.cost_function.lossFunction)
        reg_name = str(response.answer.cost_function.regularFunction)
        obj_type = str(response.answer.objectiveFunction)

        # Instantiate and link (rest of your logic remains similar)
        obj_cls = (
            self.ontology.MinObjectiveFunction
            if obj_type.lower().strip() == "minimize"
            else self.ontology.MaxObjectiveFunction
        )
        obj_instance = self._instantiate_and_format_class(
            obj_cls, f"{obj_type} Objective Function"
        )
        self._link_instances(network_instance, obj_instance, self.ontology.hasObjective)

        # Loss function handling
        known_losses = get_all_subclasses(self.ontology.LossFunction)
        best_loss_match = self._fuzzy_match_class(
            loss_name, known_losses, 90
        ) or create_subclass(self.ontology, loss_name, self.ontology.LossFunction)
        cost_instance = self._instantiate_and_format_class(
            self.ontology.CostFunction, "cost function"
        )
        loss_instance = self._instantiate_and_format_class(best_loss_match, loss_name)
        self._link_instances(obj_instance, cost_instance, self.ontology.hasCost)
        self._link_instances(cost_instance, loss_instance, self.ontology.hasLoss)

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

        self.logger.info(
            f"Processed objective functions for {network_name}: Loss: {loss_name}, Regularizer: {reg_name}, Objective: {obj_type}."
        )

    def _process_layers(self, network_instance: Thing) -> None:
        """
        Process the different layers (input, output, activation, noise, and modification) of it's network instance.
        """
        try:
            # fetch info from database
            onn = OnnxAddition()
            onn.init_engine()
            models_list = onn.fetch_models()
            num_models = len(models_list)
            prev_model = None
            subclasses: List[ThingClass] = get_all_subclasses(self.ontology.Layer)

            # fetch ann config name, find relevant model in database
            best_model_name = self._fuzzy_match_list(models_list)
            if not best_model_name:
                warnings.warn(f"Model name {best_model_name} not found in database")
                pass  # throw to josue's script for llm instantiation

            # fetch layer list of relevant model
            layer_list = onn.fetch_layers(best_model_name)

            for name in layer_list:
                layer_name, model_type, model_id, model_name = name

                # odd mismatch that is owlready2's fault, not mine
                if model_type == "Softmax":
                    model_type = "SoftMax"
                if model_type == "ReLU":  # apprently owl is very case sensitive
                    model_type = "Relu"

                best_subclass_match = self._fuzzy_match_class(
                    model_type, subclasses, 70
                )

                if (
                    not best_subclass_match
                ):  # create subclass if layer type not found in ontology
                    best_subclass_match = create_subclass(
                        self.ontology, model_type, self.ontology.Layer
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
                    best_subclass_match, layer_name
                )
                self._link_instances(
                    network_instance, layer_instance, self.ontology.hasLayer
                )

                self.logger.info(f"All layers of {model_name} successfully processed")

        except Exception as e:
            print("ERROR")
            self.logger.error(f"Error in _process_layers: {e}", exc_info=True)

    def _old_process_layers(self, network_instance: str) -> None:
        """
        Process the different layers (input, output, activation, noise, and modification) of it's network instance.
        """
        network_instance_name = self._unformat_instance_name(network_instance.name)

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
                    activation_layer_instance_name = self._unformat_instance_name(
                        activation_layer_instance.name
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
                self.logger.error(
                    "Expected an instance of Thing for Network in _process_task_characterization."
                )
                raise ValueError(
                    "Expected an instance of Thing for Network in _process_task_characterization."
                )
            if not hasattr(self.ontology, "TaskCharacterization") or not isinstance(
                self.ontology.TaskCharacterization, ThingClass
            ):
                self.logger.error(
                    "The ontology must have a valid TaskCharacterization class of type ThingClass.."
                )

            # TODO: Dynmaically provide tasks in prompt considering known task types.
            # TODO: Assumes only ones task per network, may need to change to multiple tasks.

            # Get the name of the network instance
            network_instance_name = self._unformat_instance_name(network_instance.name)

            # TODO: Find better place to put this
            general_network_header_prompt = "You are an expert in neural network architectures with deep knowledge of various models, including CNNs, RNNs, Transformers, and other advanced architectures. Your goal is to extract and provide accurate, detailed, and context-specific information about a given neural network architecture from the provided context.\n\n"
            task_characterization_prompt = f"Extract the primary task that the {network_instance_name} is designed to perform. "
            task_characterization_prompt = (
                general_network_header_prompt + task_characterization_prompt
            )  # TEMP

            task_characterization_json_format_prompt = (
                "The primary task is the most important or central objective of the network. "
                "Return the task name in JSON format with the key 'answer'.\n\n"
                # "Types of tasks include:\n"
                "Types of tasks include, choose the task that best fits:\n"
                "- Adversarial: The task of generating adversarial examples or countering another network’s predictions, often used in adversarial training or GANs. \n"
                "- Self-Supervised Classification: The task of learning useful representations without explicit labels, often using contrastive or predictive learning techniques. \n"
                "- Semi-Supervised Classification: A classification task where the network is trained on a mix of labeled and unlabeled data. \n"
                "- Supervised Classification: The task of assigning input data to predefined categories using fully labeled data. \n"
                "- Unsupervised Classification (Clustering): The task of grouping similar data points into clusters without predefined labels. \n"
                "- Discrimination: The task of distinguishing between different types of data distributions, often used in adversarial training. \n"
                "- Generation: The task of producing new data that resembles a given distribution. \n"
                # "- Reconstruction: The task of reconstructing input data, often used in denoising or autoencoders. \n"
                "Clustering: The task of grouping similar data points into clusters without predefined labels. \n"
                "- Regression: The task of predicting continuous values rather than categorical labels. \n"
                # "If the network's primary task does not fit any of the above categories, provide a conciece description of the task instead using at maximum a few words.\n\n"
                # "For example, if the network is designed to classify images of handwritten digits, the task would be 'Supervised Classification'.\n\n"
                "Expected JSON Output:\n"
                "{\n"
                '"answer": {\n'
                '"task_type": "Supervised Classification"\n'
                "}\n"
                "}\n"
            )

            task_characterization_response = self._query_llm(
                "",
                task_characterization_prompt,
                task_characterization_json_format_prompt,
                pydantic_type_schema=TaskCharacterizationResponse,
            )

            if not task_characterization_response:
                self.logger.warning(
                    f"No response for task characterization for network instance '{network_instance_name}'."
                )
                return

            # Extract the task type from the response
            task_type_name = str(task_characterization_response.answer.task_type)

            # Get all known task types for TaskCharacterization
            known_task_types = get_all_subclasses(self.ontology.TaskCharacterization)

            if not known_task_types:
                self.logger.warning(
                    f"No known task types found in the ontology, creating a new task type for {task_type_name} in the {network_instance_name}."
                )
                best_match_task_type = create_subclass(
                    self.ontology, task_type_name, self.ontology.TaskCharacterization
                )
            else:
                # Check if the task type matches any known task types
                best_match_task_type = self._fuzzy_match_class(
                    task_type_name, known_task_types, 90
                )
                if not best_match_task_type:
                    best_match_task_type = create_subclass(
                        self.ontology,
                        task_type_name,
                        self.ontology.TaskCharacterization,
                    )

            # Instantiate and link the task characterization instance with the network instance
            task_type_instance = self._instantiate_and_format_class(
                best_match_task_type, task_type_name
            )
            self.logger.info(
                f"Processed task characterization '{task_type_name}', linked to network instance '{network_instance_name}."
            )

            self._link_instances(
                network_instance, task_type_instance, self.ontology.hasTaskType
            )
        except Exception as e:
            self.logger.error(
                f"Error processing task characerization for network instance '{network_instance_name}': {e}",
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
                logger.error("Invalid or missing Network class in the ontology.", exc_info=True)
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
                # self._process_layers(network_instance) # May be processed by onnx
                self._process_objective_functions(network_instance)
                # self._process_task_characterization(network_instance)
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

    def _process_dataset(self, dataset_pipe_instance: Thing) -> None:
        """
        Process the dataset class and it's components.
        """
        try:
            if not dataset_pipe_instance:
                self.logger.error(
                    "No DatasetPipe instance provided in _process_dataset."
                )
                raise ValueError("No DatasetPipe instance in the ontology.")

            self.logger.info("Starting to process dataset.")
            dataset_instance = self._instantiate_and_format_class(
                self.ontology.Dataset, "Dataset"
            )
            self._link_instances(
                dataset_pipe_instance, dataset_instance, self.ontology.joinsDataSet
            )

            # Define dataset properties and datatype

            dataset_prompt = "Based on the provided paper, extract and describe the dataset details used.\n"

            dataset_json_format_prompt = (
                "Your answer should include the following information:\n"
                "- data_description: A brief description of the dataset, including what data it contains, its source, and the number of examples. (Required)\n"
                "- data_doi: The DOI of the dataset, if available. (Optional)\n"
                "- data_location: The physical or digital location of the dataset. (Optional)\n"
                '- data_sample_dimensionality: The dimensions or shape of a single data sample (e.g., "28x28" for MNIST images). (Optional)\n'
                "- data_sample_features: A description of the features or attributes present in each data sample. (Optional)\n"
                "- data_samples: The total number of data samples in the dataset. (Optional)\n"
                "- is_transient_dataset: A boolean indicating whether the dataset is transient (temporary) or persistent. (Optional)\n"
                '- dataType: An object with a key "subclass" representing the type of data present in the dataset. '
                'The subclass must be one of the following: "Image", "MultiDimensionalCube", "Text", or "Video".\n\n'
                'Return your answer strictly in JSON format with a key "answer". For example:\n'
                "{\n"
                '  "answer": {\n'
                '    "data_description": "The MNIST database of handwritten digits, containing 60,000 training and 10,000 test examples.",\n'
                '    "data_doi": "10.1234/mnist",\n'
                '    "data_location": "http://yann.lecun.com/exdb/mnist/",\n'
                '    "data_sample_dimensionality": "28x28",\n'
                '    "data_sample_features": "Grayscale pixel values",\n'
                '    "data_samples": 70000,\n'
                '    "is_transient_dataset": false,\n'
                '    "dataType": {\n'
                '      "subclass": "Image"\n'
                "    }\n"
                "  }\n"
                "}\n"
                "If any optional field is not available, you may omit it or return None."
            )

            dataset_response = self._query_llm(
                "",
                dataset_prompt,
                dataset_json_format_prompt,
                pydantic_type_schema=DatasetResponse,
            )
            if not dataset_response:
                self.logger.warning("No response received for dataset details.")
                return

            dataset_details = dataset_response.answer

            dataset_instance.data_description = [dataset_details.data_description]
            if dataset_details.data_doi is not None:
                dataset_instance.data_doi = [dataset_details.data_doi]
            if dataset_details.data_location is not None:
                dataset_instance.data_location = [dataset_details.data_location]
            if dataset_details.data_sample_dimensionality is not None:
                dataset_instance.data_sample_dimensionality = [
                    dataset_details.data_sample_dimensionality
                ]
            if dataset_details.data_sample_features is not None:
                dataset_instance.data_sample_features = [
                    dataset_details.data_sample_features
                ]
            if dataset_details.data_samples is not None:
                dataset_instance.data_samples = [dataset_details.data_samples]
            if dataset_details.is_transient_dataset is not None:
                dataset_instance.is_transient_dataset = [
                    dataset_details.is_transient_dataset
                ]

            # Process Datatype
            best_data_type_match = self._fuzzy_match_class(
                dataset_details.dataType.subclass,
                get_all_subclasses(self.ontology.DataType),
            )
            if best_data_type_match:
                self.logger.info(f"Best Data Type Match: {best_data_type_match}")
                data_type_instance = self._instantiate_and_format_class(
                    best_data_type_match, "Data Type"
                )
                self._link_instances(
                    dataset_instance, data_type_instance, self.ontology.hasDataType
                )
            else:
                self.logger.info(
                    f"Unknown Data Type: {dataset_details.dataType.subclass}"
                )

            self.logger.info("Finished processing dataset.")

            # print(f"Dataset Details: {dataset_details}")
            # print(f"Dataset Details Data Description: {dataset_details.data_description}")
            # print(f"Dataset Details Data DOI: {dataset_details.data_doi}")
            # print(f"Dataset Details Data Location: {dataset_details.data_location}")
            # print(f"Dataset Details Data Sample Dimensionality: {dataset_details.data_sample_dimensionality}")
            # print(f"Dataset Details Data Sample Features: {dataset_details.data_sample_features}")
            # print(f"Dataset Details Data Samples: {dataset_details.data_samples}")
            # print(f"Dataset Details Is Transient Dataset: {dataset_details.is_transient_dataset}")
            # print(f"Dataset Details Data Type: {dataset_details.dataType.subclass}")

            # print(f"Dataset Instance: {dataset_instance}")
            # print(f"Dataset Instance Data Description: {dataset_instance.data_description}")
            # print(f"Dataset Instance Data DOI: {dataset_instance.data_doi}")
            # print(f"Dataset Instance Data Location: {dataset_instance.data_location}")
            # print(f"Dataset Instance Data Sample Dimensionality: {dataset_instance.data_sample_dimensionality}")
            # print(f"Dataset Instance Data Sample Features: {dataset_instance.data_sample_features}")
            # print(f"Dataset Instance Data Samples: {dataset_instance.data_samples}")
            # print(f"Dataset Instance Is Transient Dataset: {dataset_instance.is_transient_dataset}")
            # print(f"Dataset Instance Data Type: {dataset_instance.hasDataType}")
        except Exception as e:
            self.logger.error(f"Error in _process_dataset: {e}", exc_info=True)
            raise e

    def _process_training_strategy(self, ann_config_instance: Thing) -> None:
        """
        Process training strategy for an Ann Configuration instance.
        """
        try:
            if not ann_config_instance:
                self.logger.error(
                    "No ANN Configuration instance provided in _process_training_strategy."
                )
                raise ValueError("No ANN Configuration instance in the ontology.")

            if hasattr(self.ontology, "TrainingStrategy"):
                # Instantiate the training strategy instance with name of the ANN Configuration instance _training-strategy
                training_strategy_instance = self._instantiate_and_format_class(
                    self.ontology.TrainingStrategy, "Training Strategy"
                )
                self._link_instances(
                    ann_config_instance,
                    training_strategy_instance,
                    self.ontology.hasPrimaryTrainingSession,
                )

                # Instantiate primary training session
                training_session_instance = self._instantiate_and_format_class(
                    self.ontology.TrainingSession, "Training Session"
                )
                self._link_instances(
                    training_strategy_instance,
                    training_session_instance,
                    self.ontology.hasPrimaryTrainingSession,
                )

                # Instantiate TrainingStep subclass: 'NetworkSpecific' with subclass 'Training Single'
                training_single_instance = self._instantiate_and_format_class(
                    self.ontology.TrainingSingle, "Training Single"
                )

                # network_specific_instance = self._instantiate_and_format_class(self.ontology.NetworkSpecific, "Network Specific")
                self._link_instances(
                    training_session_instance,
                    training_single_instance,
                    self.ontology.hasPrimaryTrainingStep,
                )

                training_single_prompt = "Based on the provided context, extract the training details used in the network-specific training step."
                # JSON format instructions: How the JSON output should be structured.
                training_single_json_format_prompt = (
                    "Return a JSON object with a key 'answer'. The value should be an object with the following keys:\n"
                    "- batch_size: an integer representing the number of examples per iteration.\n"
                    "- learning_rate_decay: a float representing the rate at which the learning rate decays.\n"
                    "- number_of_epochs: an integer representing the total number of epochs.\n"
                    "- learning_rate_decay_epochs (optional): an integer representing the epoch at which decay is applied, if available.\n\n"
                    "For example, if the training step uses a batch size of 32, a learning rate decay of 0.01, "
                    "number of epochs 10, and a learning_rate_decay_epochs of 5, the output should look like:\n"
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

                # Training Single Response should be in pydantic format
                training_single_response = self._query_llm(
                    "",
                    training_single_prompt,
                    training_single_json_format_prompt,
                    pydantic_type_schema=TrainingSingleResponse,
                )
                if not training_single_response:
                    self.logger.warning("No response received for training details.")
                    return

                training_single_details = training_single_response.answer
                training_single_instance.batch_size = [
                    training_single_details.batch_size
                ]
                training_single_instance.learning_rate_decay = [
                    training_single_details.learning_rate_decay
                ]
                training_single_instance.number_of_epochs = [
                    training_single_details.number_of_epochs
                ]
                # Learning rate is optional
                if training_single_details.learning_rate_decay_epochs is not None:
                    training_single_instance.learning_rate_decay_epochs = [
                        training_single_details.learning_rate_decay_epochs
                    ]

                # print(f"Training Single Details: {training_single_details}")
                # print(f"Training Single Instance: {training_single_instance}")
                # print(f"Training Single Instance Batch Size: {training_single_instance.batch_size}")
                # print(f"Training Single Instance Learning Rate Decay: {training_single_instance.learning_rate_decay}")
                # print(f"Training Single Instance Number of Epochs: {training_single_instance.number_of_epochs}")
                # print(f"Training Single Instance Learning Rate Decay Epochs: {training_single_instance.learning_rate_decay_epochs}")

                # Process TrainingSingle connected classes:

                # Instantiate DatasetPipe for primary training session
                dataset_pipe_instance = self._instantiate_and_format_class(
                    self.ontology.DatasetPipe, "Dataset Pipe"
                )
                self._link_instances(
                    training_single_instance,
                    dataset_pipe_instance,
                    self.ontology.trainingSingleHasIOPipe,
                )

                # Instantiate dataset for dataset pipe
                dataset_instance = self._instantiate_and_format_class(
                    self.ontology.Dataset, "Dataset"
                )
                self._link_instances(
                    dataset_pipe_instance, dataset_instance, self.ontology.joinsDataSet
                )

                # Process dataset
                self._process_dataset(dataset_pipe_instance)
                self.logger.info("Finished processing training strategy.")
            else:
                self.logger.error("TrainingStrategy class not found in the ontology.")
                raise AttributeError(
                    "TrainingStrategy class not found in the ontology."
                )

        except:
            self.logger.error(
                f"Error in _process_training_strategy: {e}", exc_info=True
            )
            raise e

    def save_ontology(self) -> None:
        """
        Saves the ontology to the pre-specified file.
        """
        self.ontology.save(file=self.output_owl_path, format="rdfxml")
        self.logger.info(f"Ontology saved to {self.output_owl_path}")

    def run(self, ann_path:List[str]) -> None:
        """
        Main method to run the ontology instantiation process.
        """
        try:
            with self.ontology:
                start_time = time.time()

                if not hasattr(self.ontology, "ANNConfiguration"):
                    raise AttributeError(
                        "Error: Class 'ANNConfiguration' not found in ontology.", exc_info=True
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

                create_class_data_property(self.ontology, "hasTitle", self.ontology.ANNConfiguration, str, True)
                for title in grab_titles():
                    ann_config_instance.hasTitle = title
                create_class_data_property(self.ontology, "hasPaperPath", self.ontology.ANNConfiguration, str, True)
                for path in ann_path:
                    ann_config_instance.hasPaperPath = path


                # Process the network class and it's components.
                self._process_network(ann_config_instance)

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


def instantiate_annetto(
    ann_name: str, ann_path: str, ontology: Ontology, ontology_output_path: str
) -> None:
    """
    Instantiates an ANN ontology from the provided ANN Configuration filepath.
    Papers and Code must be extracted to the proper JSON format beforehand.

    :param: ann_name: The name of the ANN Configuration.
    :param: ann_path: The path to the ANN Configuration JSON files.
    :param: ontology: The ontology to instantiate.
    :param: ontology_output_path: The path to save the ontology.
    """
    list_json_doc_paths = glob.glob(f"{ann_path}/*.json") # TODO: Lazy to just glob all json files in the directory
    list_pdf_paths = glob.glob(f"{ann_path}/*.pdf")
    instantiator = OntologyInstantiator(
        list_json_doc_paths, ann_name, ontology=ontology, output_owl_path=ontology_output_path
    )
    instantiator.run(list(list_pdf_paths))
    instantiator.save_ontology()
    print(f"Ontology instantiation completed for {ann_name} and saved to {ontology_output_path}.")


# For standalone testing
if __name__ == "__main__":
    import glob
    time_start = time.time()

    for model_name in [
        # "alexnet",
        "resnet",
        # "vgg16",
        # "gan", # Assume we can model name from user or something
    ]:
        instantiate_annetto(
                model_name,
                f"data/{model_name}",
                load_annetto_ontology("base"),
                C.ONTOLOGY.TEST_ONTOLOGY_PATH,
            )
        # try:
        #     instantiate_annetto(
        #         model_name,
        #         f"data/{model_name}",
        #         load_annetto_ontology("base"),
        #         C.ONTOLOGY.TEST_ONTOLOGY_PATH,
        #     )
        # except Exception as e:
        #     print(f"Error instantiating the {model_name} ontology in __name__: {e}")
        #     continu
        
    time_end = time.time()
    minutes, seconds = divmod(time_end - time_start, 60)
    print(f"Total time taken: {int(minutes)} minutes and {seconds:.2f} seconds.")
