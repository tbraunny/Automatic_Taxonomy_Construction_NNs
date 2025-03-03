import logging
import os
import hashlib
from datetime import datetime

from typing import Dict, Any, Union, List, Optional
from owlready2 import Ontology, ThingClass, Thing, ObjectProperty, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    create_cls_instance,
    assign_object_property_relationship,
    create_subclass,
    get_all_subclasses,
    get_class_by_name
)
from utils.annetto_utils import int_to_ordinal, make_thing_classes_readable
from utils.llm_service import init_engine, query_llm
from rapidfuzz import process, fuzz
from utils.onnx_additions.add_onnx import OnnxAddition
from utils.annetto_utils import fuzzy_match_class

# TOMS VERSION

try:
    import tiktoken
except ImportError:
    tiktoken = None


OMIT_CLASSES = {
    "DataCharacterization",
    "Regularization",
}  # Classes to omit from instantiation.


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

def setup_logger():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"ann_config_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")

    logger = logging.getLogger("LLMServiceLogger")
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to prevent duplicate logging
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Use the dedicated logger
logger = setup_logger()


class OntologyInstantiator:
    """
    A class to instantiate an annett-o ontology by processing each main component separately and linking them together.
    """

    def __init__(
        self, ontology: Ontology, json_file_path: str, ann_config_name: str = "AlexNet"
    ) -> None:
        self.ontology = ontology
        self.json_file_path = json_file_path
        self.llm_cache: Dict[str, Any] = {}
        self.logger = logger
        self.ann_config_name = ann_config_name  # Assume AlexNet for now.
        self.ann_config_hash = self._generate_hash(self.ann_config_name)

    def _generate_hash(self, str: str) -> str:
        """
        Generate a unique hash identifier based on the given string.
        """
        hash_object = hashlib.md5(
            str.encode()
        )  # Generate a consistent hash
        return hash_object.hexdigest()[:8]

    def _instantiate_cls(self, cls: ThingClass, instance_name: str) -> Thing:
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

    def _fuzzy_match_class(
        self, instance_name: str, classes: List[ThingClass], threshold: int = 80
    ) -> Optional[ThingClass]:
        """
        Perform fuzzy matching to find the best match for an instance to a known class.

        :param instance_name: The instance name.
        :param classes: A list of ThingClass objects to match with.
        :param threshold: The minimum score required for a match.
        :return: The best-matching ThingClass object or None if no good match is found.
        """
        # for cls in classes:
        #     print(f"Class: {cls}, Name: {cls.name}, IRI: {cls.iri}")

        if not instance_name or not classes:
            return None

        # Convert classes to a dictionary for lookup
        class_name_map = {cls.name: cls for cls in classes}

        match, score, _ = process.extractOne(
            instance_name, class_name_map.keys(), scorer=fuzz.ratio
        )

        return class_name_map[match] if score >= threshold else None

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
        self, instructions: str, prompt: str
    ) -> Union[Dict[str, Any], int, str, List[str]]:
        """
        Queries the LLM with a structured prompt to obtain a response in a specific JSON format.

        The prompt should include few-shot examples demonstrating the expected structure of the output.
        The LLM is expected to return a JSON object where the primary key is `"answer"`, and the value
        can be one of the following types:
        - Integer (e.g., `{"answer": 100}`)
        - String (e.g., `{"answer": "ReLU Activation"}`)
        - List of strings (e.g., `{"answer": ["L1 Regularization", "Dropout"]}`)
        - Dictionary mapping strings to integers (e.g., `{"answer": {"Convolutional": 4, "FullyConnected": 1}}`)

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

        Returns:
            Union[dict, int, str, list[str]]: The parsed LLM response based on the provided examples.
        """
        full_prompt = f"{instructions}\n{prompt}"
        if full_prompt in self.llm_cache:
            self.logger.info(f"Using cached LLM response for prompt: {full_prompt}")
            print("Using cached LLM response #####")
            return self.llm_cache[full_prompt]
        try:
            response = query_llm(self.ann_config_name, full_prompt)

            self.logger.info(f"LLM query: {full_prompt}")
            self.logger.info(f"LLM query response: {response}")
            self.llm_cache[full_prompt] = response

            return response
        except Exception as e:
            self.logger.error(f"LLM query error: {e}")
            return ""

    def _process_objective_functions(self, network_instance: Thing) -> None:
        """
        Process loss and regularizer functions, and link them to it's network instance.
        """
        try:
            network_instance_name = self._unhash_and_format_instance_name(network_instance.name)

            # Gets a list of all subclasses of LossFunction in a readable format, used for few shot exampling.
            loss_function_subclass_names = make_thing_classes_readable(
                get_all_subclasses(self.ontology.LossFunction)
            )

            loss_function_prompt = (
                f"Step 1: Identify and extract only the loss function names explicitly used in the {network_instance_name}'s architecture. "
                f"Consider only those loss functions that are directly part of {network_instance_name}.\n\n"

                "Step 2: For each extracted loss function, determine whether it is designed to **minimize** or **maximize** its objective function.\n"
                "- If a function minimizes an error, cost, or divergence (e.g., Cross-Entropy Loss, MSE, Huber Loss), classify it as **minimize**.\n"
                "- If a function maximizes a likelihood, reward, or score function (e.g., Log-Likelihood, Reinforcement Learning Reward Maximization), classify it as **maximize**.\n"
                "- If a function minimizes the negative of a quantity (e.g., negative log-likelihood), it is still a **minimization problem**.\n"
                "- Carefully analyze each function before making a decision.\n\n"

                """Step 3: After reasoning through the loss functions and their objectives, return the final structured output **strictly in JSON format** using "answer" as the key", following this exact format:\n\n"""

                "**Expected JSON Format:**\n"
                '{"answer": [{"loss_function": "Loss Function Name", "objective": "minimize or maximize"}]}\n\n'

                f"Some examples of loss functions include {loss_function_subclass_names}. If one of these match, response with its exact name\n\n"

                "**Format Examples:**\n"
                "Network: Discriminator\n"
                '{"answer": [{"loss_function": "Binary Cross-Entropy Loss", "objective": "minimize"}]}\n'
                "Network: Discriminator\n"
                '{"answer": [{"loss_function": "Wasserstein Loss", "objective": "minimize"}, {"loss_function": "Hinge Loss", "objective": "minimize"}]}\n'
                "Network: Reinforcement Learning Model\n"
                '{"answer": [{"loss_function": "Policy Gradient Loss", "objective": "maximize"}]}\n\n'
                "Network: Generator\n"
                '{"answer": []\n\n'

                f"Now, analyze the following network and think through the response carefully:\n"
                f"Network: {network_instance_name}\n\n"

                "**DO NOT OUTPUT JSON UNTIL YOU HAVE FINISHED THINKING.**\n"
                "Once you have completed your reasoning, format your final answer strictly as JSON:\n\n"
                '{"answer": "<Your Answer Here>"}'
            )
            loss_function_names_and_objective = self._query_llm("", loss_function_prompt)


            # If no loss function names are provided, create a default loss function instance and return.
            if not loss_function_names_and_objective:
                self.logger.warning(f"No response for loss function in network {network_instance_name}.")

                loss_name = "Unknown Loss Function"
                cost_function_instance = self._instantiate_cls(
                    self.ontology.CostFunction, "cost function"
                )
                loss_function_instance = self._instantiate_cls(
                    self.ontology.LossFunction, loss_name
                )
                objective_function_instance = self._instantiate_cls(
                    self.ontology.MinObjectiveFunction, "Unknown Objective Function"
                )
                
                self._link_instances(
                    objective_function_instance,
                    cost_function_instance,
                    self.ontology.hasCost,
                )
                self._link_instances(
                    cost_function_instance, loss_function_instance, self.ontology.hasLoss
                )
            
            # Iterate through dictionary of loss function names and their objectives
            for loss_function_name, objective_type in loss_function_names_and_objective.items():

                # Instantiate the objective function based on the objective type
                if objective_type.lower() == "minimize":
                    objective_function_instance = self._instantiate_cls(
                        self.ontology.MinObjectiveFunction, "Min Objective Function"
                    )
                elif objective_type.lower() == "maximize":
                    objective_function_instance = self._instantiate_cls(
                        self.ontology.MaxObjectiveFunction, f"Max Objective Function"
                    )
                else:
                    self.logger.warning(
                        f"Invalid response for loss function objective type for {loss_function_name}, using minimzie as default.")
                    objective_function_instance = self._instantiate_cls(
                        self.ontology.MinObjectiveFunction, "Min Objective Function"
                    ) # Default to minimize if no response

                # Get all known loss functions for the loss function
                known_loss_functions = get_all_subclasses(self.ontology.Task)

                # Check if the loss function name matches any known loss function
                best_match_loss_class = self._fuzzy_match_class(loss_function_name, known_loss_functions, 95)
                if not best_match_loss_class:
                    best_match_loss_class = create_subclass(self.ontology.LossFunction, loss_function_name)

                # Instantiate the cost function and loss function
                cost_function_instance = self._instantiate_cls(
                    self.ontology.CostFunction, "cost function"
                )
                loss_function_instance = self._instantiate_cls(
                    best_match_loss_class, loss_function_name
                )

                # Link the objective function to the cost function and the cost function to the loss function
                self._link_instances(
                    objective_function_instance,
                    cost_function_instance,
                    self.ontology.hasCost,
                )
                self._link_instances(
                    cost_function_instance, loss_function_instance, self.ontology.hasLoss
                )

                regularizer_function_prompt = (
                    f"Extract only the names of explicit regularizer functions that are mathematically added to the objective function for the {loss_name} loss function. "
                    "Exclude implicit regularization techniques like Dropout, Batch Normalization, or any regularization that is not directly part of the loss function. "
                    "Return the result in JSON format with the key 'answer'. Follow the examples below.\n\n"

                    "Clarifications:\n"
                    "L2 Regularization is a technique that explicitly adds a penalty term to the loss function, encouraging smaller weights and reducing overfitting. This means that during backpropagation, the gradient update includes an additional term derived from this penalty."
                    "Weight Decay is an optimization technique that directly modifies the weight update rule by scaling the weights down after each step, effectively implementing L2 regularization but without explicitly altering the loss function"
                    "\n\n"

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
                    self.logger.info(
                        f"No response for regularizer function classes for loss function {loss_name}."
                    )
                    continue
                    
                for reg_name in regularizer_names:
                    best_match_reg_class = self._fuzzy_match_class(regularizer_names, known_loss_functions, 95)

                    if not best_match_reg_class:
                        best_match_class = create_subclass(self.ontology.RegularizerFunction, reg_name)
                    
                    reg_instance = self._instantiate_cls(
                        best_match_class, reg_name
                    )
                    self._link_instances(
                        cost_function_instance, reg_instance, self.ontology.hasRegularizer
                    )
        except Exception as e:
            self.logger.error(f"Error processing objective functions: {e}")

            loss_name = "Unknown Loss Function"
            cost_function_instance = self._instantiate_cls(
                self.ontology.CostFunction, "cost function"
            )
            loss_function_instance = self._instantiate_cls(
                self.ontology.LossFunction, loss_name
            )
            objective_function_instance = self._instantiate_cls(
                self.ontology.MinObjectiveFunction, "Unknown Objective Function"
            )
            
            self._link_instances(
                objective_function_instance,
                cost_function_instance,
                self.ontology.hasCost,
            )
            self._link_instances(
                cost_function_instance, loss_function_instance, self.ontology.hasLoss
            )
            return
            
            
        # loss_function_prompt = (
        #     f"Extract only the names of the loss functions used for only the {network_instance_name}'s architecture and return the result in JSON format with the key 'answer'. "
        #     f"Examples of loss functions include {loss_function_subclass_names}."
        #     "Follow the examples below.\n\n"
        #     "Examples:\n"
        #     "Network: Discriminator\n"
        #     '{"answer": ["Binary Cross-Entropy Loss"]}\n'
        #     "Network: Discriminator\n"
        #     '{"answer": ["Wasserstein Loss", "Hinge Loss"]}\n\n'
        #     f"Now, for the following network:\nNetwork: {network_instance_name}\n"
        #     '{"answer": "<Your Answer Here>"}'
        # )
        # loss_function_names = self._query_llm("", loss_function_prompt)

        # # If no loss function names are provided, create a default loss function instance and return.
        # if not loss_function_names:
        #     self.logger.info("No response for loss function classes.")
            
        #     loss_name = "Unknown Loss Function"

        #     cost_function_instance = self._instantiate_cls(
        #         self.ontology.CostFunction, "cost function"
        #     )
        #     loss_function_instance = self._instantiate_cls(
        #         self.ontology.LossFunction, loss_name
        #     )
        #     self._link_instances(
        #         objective_function_instance,
        #         cost_function_instance,
        #         self.ontology.hasCost,
        #     )
        #     self._link_instances(
        #         cost_function_instance, loss_function_instance, self.ontology.hasLoss
        #     )
        #     return

        # for loss_name in loss_function_names:
        #     loss_objective_prompt = (
        #         f"Is the {loss_name} function designed to minimize or maximize its objective function? "
        #         "Please respond with either 'minimize' or 'maximize' in JSON format using the key 'answer'.\n\n"
        #         "**Clarification:**\n"
        #         "- If the function is set up to minimize (e.g., cross-entropy, MSE), respond with 'minimize'.\n"
        #         "- If the function is set to maximize a likelihood or score function (e.g., log-likelihood, accuracy), respond with 'maximize'.\n"
        #         "- Note: Maximizing the log-probability typically corresponds to minimizing the negative log-likelihood.\n\n"
        #         "Examples:\n"
        #         "Loss Function: Cross-Entropy Loss\n"
        #         '{"answer": "minimize"}\n'
        #         "Loss Function: Custom Score Function\n"
        #         '{"answer": "maximize"}\n\n'
        #         f"Now, for the following loss function:\nLoss Function: {loss_name}\n"
        #         '{"answer": "<Your Answer Here>"}'
        #     )
        #     loss_obj_response = self._query_llm("", loss_objective_prompt)
        #     if not loss_obj_response:
        #         self.logger.info(
        #             f"No response for loss function objective for {loss_name}."
        #         )
        #         # Assume minimize if no response.
        #         loss_obj_response = "minimize"
        #     loss_obj_type = loss_obj_response.lower()

        #     if loss_obj_type == "minimize":
        #         objective_function_instance = self._instantiate_cls(
        #             self.ontology.MinObjectiveFunction, "Min Objective Function"
        #         )
        #     elif loss_obj_type == "maximize":
        #         objective_function_instance = self._instantiate_cls(
        #             self.ontology.MaxObjectiveFunction, f"Max Objective Function"
        #         )
        #     else:
        #         self.logger.info(
        #             f"Invalid response for loss function objective for {loss_name}."
        #         )
        #         continue

        #     cost_function_instance = self._instantiate_cls(
        #         self.ontology.CostFunction, "cost function"
        #     )
        #     loss_function_instance = self._instantiate_cls(
        #         self.ontology.LossFunction, loss_name
        #     )

        #     self._link_instances(
        #         objective_function_instance,
        #         cost_function_instance,
        #         self.ontology.hasCost,
        #     )
        #     self._link_instances(
        #         cost_function_instance, loss_function_instance, self.ontology.hasLoss
        #     )

        #     regularizer_function_prompt = (
        #         f"Extract only the names of explicit regularizer functions that are mathematically added to the objective function for the {loss_name} loss function. "
        #         "Exclude implicit regularization techniques like Dropout, Batch Normalization, or any regularization that is not directly part of the loss function. "
        #         "Return the result in JSON format with the key 'answer'. Follow the examples below.\n\n"

        #         "Clarifications:\n"
        #         "L2 Regularization is a technique that explicitly adds a penalty term to the loss function, encouraging smaller weights and reducing overfitting. This means that during backpropagation, the gradient update includes an additional term derived from this penalty."
        #         "Weight Decay is an optimization technique that directly modifies the weight update rule by scaling the weights down after each step, effectively implementing L2 regularization but without explicitly altering the loss function"
        #         "\n\n"

        #         "Examples:\n"
        #         "Loss Function: Discriminator Loss\n"
        #         '{"answer": ["L1 Regularization"]}\n\n'
        #         "Loss Function: Generator Loss\n"
        #         '{"answer": ["L2 Regularization", "Elastic Net"]}\n\n'
        #         "Loss Function: Cross-Entropy Loss\n"
        #         '{"answer": []}\n\n'
        #         "Loss Function: Binary Cross-Entropy Loss\n"
        #         '{"answer": ["L2 Regularization"]}\n\n'
        #         f"Now, for the following loss function:\nLoss Function: {loss_name}\n"
        #         '{"answer": "<Your Answer Here>"}'
        #     )
        #     regularizer_names = self._query_llm("", regularizer_function_prompt)
        #     if not regularizer_names:
        #         self.logger.info(
        #             f"No response for regularizer function classes for loss function {loss_name}."
        #         )
        #         continue
        #     if regularizer_names == []:
        #         self.logger.info(
        #             f"No regularizer functions provided for loss function {loss_name}."
        #         )
        #         continue

        #     for reg_name in regularizer_names:
        #         reg_instance = self._instantiate_cls(
        #             self.ontology.RegularizerFunction, reg_name
        #         )
        #         self._link_instances(
        #             cost_function_instance, reg_instance, self.ontology.hasRegularizer
        #         )

    def process_layers(self, network_instance: str) -> None:
        """
        Process the different layers (input, output, activation, noise, and modification) of it's network instance.
        """
        # fetch info from database
        onn = OnnxAddition()
        onn.init_engine()
        layer_list , model_list = onn.fetch_layers()
        num_models = len(model_list)
        prev_model = None

        for name in layer_list:
            layer_name , model_type , model_id , attributes = name

            model_str = str(model_id) # for cls instantiation compatibility
            ann_config = self._instantiate_cls(self.ontology.ANNConfiguration, model_str)

            if not hasattr(self.ontology.Network , model_str): # prevent duplicate networks
                network_instance = self._instantiate_cls(self.ontology.Network , model_str)

            # odd mismatch that is owlready2's fault, not mine
            if model_type == "Softmax":
                model_type = "SoftMax"
            if model_type == "ReLU": # apprently owl is very case sensitive
                model_type = "Relu"

            self._link_instances(ann_config , network_instance , self.ontology.hasNetwork)

            subclasses = get_all_subclasses(self.ontology.Layer)
            best_subclass_match = self._fuzzy_match_class(model_type , subclasses , 95)

            if not best_subclass_match: # create subclass if layer type not found in ontology
                best_subclass_match = create_subclass(self.ontology , model_type , self.ontology.Layer)
            
            #Debugging
            if model_id != prev_model:
                print(f"Processing model {model_id} / {num_models}" , end='\r')
                #self.logger.info(f"Model ID: " , model_id , "\nSubclass: " , best_subclass_match , "\nModel Type: " , model_type , "\n Match Type: " , type(best_subclass_match))
                # if isinstance(best_subclass_match, Thing):
                #     self.logger.info(f"{best_subclass_match} is an instance of Thing.")
                # else:
                #     self.logger.info(f"{best_subclass_match} is not an instance of Thing.") 

            layer_instance = self._instantiate_cls(best_subclass_match , layer_name)
            self._link_instances(network_instance , layer_instance , self.ontology.hasLayer)

        self.logger.info("Finished processing layers")
            

    def _process_task_characterization(self, network_instance: Thing) -> None:
        """
        Process the task characterization of a network instance.
        """
        if not network_instance:
            raise ValueError("No network instance found in the full context.")

        network_instance_name = self._unhash_and_format_instance_name(network_instance.name)

        # Prompt LLM to extract task name
        task_prompt = (
            f"Extract the primary task that the {network_instance_name} network architecture is designed to perform. "
            "The primary task is the most important or central objective of the network. "
            "Return the task name in JSON format with the key 'answer'.\n\n"
            "Examples of types of tasks include:\n"
            "- **Adversarial**: The task of generating adversarial examples or countering another network’s predictions, often used in adversarial training or GANs. \n"
            "  Example: A model that generates images to fool a classifier.\n\n"
            "- **Self-Supervised Classification**: The task of learning useful representations without explicit labels, often using contrastive or predictive learning techniques. \n"
            "  Example: A network pre-trained using contrastive learning and later fine-tuned for classification.\n\n"
            "- **Semi-Supervised Classification**: A classification task where the network is trained on a mix of labeled and unlabeled data. \n"
            "  Example: A model trained with a small set of labeled images and a large set of unlabeled ones for better generalization.\n\n"
            "- **Supervised Classification**: The task of assigning input data to predefined categories using fully labeled data. \n"
            "  Example: A CNN trained on labeled medical images to classify diseases.\n\n"
            "- **Unsupervised Classification (Clustering)**: The task of grouping similar data points into clusters without predefined labels. \n"
            "  Example: A model that clusters news articles into topics based on similarity.\n\n"
            "- **Discrimination**: The task of distinguishing between different types of data distributions, often used in adversarial training. \n"
            "  Example: A discriminator in a GAN that differentiates between real and generated images.\n\n"
            "- **Generation**: The task of producing new data that resembles a given distribution. \n"
            "  Example: A generative model that creates realistic human faces from random noise.\n\n"
            "- **Reconstruction**: The task of reconstructing input data, often used in denoising or autoencoders. \n"
            "  Example: A model that removes noise from images to restore the original content.\n\n"
            "- **Regression**: The task of predicting continuous values rather than categorical labels. \n"
            "  Example: A neural network that predicts house prices based on features like size and location.\n\n"
            "If the network's primary task does not fit any of the above categories, provide a conciece description of the task instead using at maximum a few words.\n\n"
            "JSON output examples:\n"
            "1. Network: Discriminator\n"
            '{"answer": "Discrimination"}\n\n'
            "2. Network: Generator\n"
            '{"answer": "Generation"}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": "Regression"}\n\n'
            f"Now, for the following network:\nNetwork: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )

        task_name = self._query_llm("", task_prompt)
        task_name = task_name.lower().strip()

        if not task_name:
            self.logger.warning("No response for task characterization.")
            task_name = "Unknown Task"

        # Get subclasses of TaskCharacterization
        known_tasks_classes = get_all_subclasses(self.ontology.TaskCharacterization)
        
        # Perform fuzzy matching
        best_match_task_class = self._fuzzy_match_class(task_name, known_tasks_classes, 95)
        if not best_match_task_class:
            best_match_task_class = create_subclass(
                self.ontology, task_name, self.ontology.TaskCharacterization
            )

        # Instantiate and link the task characterization instance
        task_class = get_class_by_name(self.ontology, task_name)
        task_instance = self._instantiate_cls(task_class, task_name)
        self._link_instances(network_instance, task_instance, self.ontology.hasTaskType)

        self.logger.info(f"Task characterization '{task_name}' linked to network instance '{network_instance_name}'.")

    def _process_network(self, ann_config_instance: Thing) -> None:
        """
        Process the network class and it's components.
        """
        if not ann_config_instance:
            logger.error("No ANN Configuration instance in the ontology.")
            raise ValueError("No ANN Configuration instance in the ontology.")

        if hasattr(self.ontology, "Network"):

            network_instances = []

            if (
                self._unhash_and_format_instance_name(ann_config_instance.name) == "gan"
            ):  # Temp for gan & multi network
                network_instances.append(
                    self._instantiate_cls(self.ontology.Network, "Generator Network")
                )
                network_instances.append(
                    self._instantiate_cls(
                        self.ontology.Network, "Discriminator Network"
                    )
                )

                # Process the components of the network instance.
                for network_instance in network_instances:
                    self._link_instances(
                        ann_config_instance, network_instance, self.ontology.hasNetwork
                    )
                    # self._process_layers(network_instance) # May be processed by onnx
                    self._process_objective_functions(network_instance)
                    # self._process_task_characterization(network_instance)
            else:
                # Here is where logic for processing the network instance would go.
                network_instances.append(
                    self._instantiate_cls(
                        self.ontology.Network, "Convolutional Network"
                    )
                )  # assumes network is convolutional for cnn

                # Process the components of the network instance.
                for network_instance in network_instances:
                    self._link_instances(
                        ann_config_instance, network_instance, self.ontology.hasNetwork
                    )
                    # self._process_layers(network_instance) # May be processed by onnx
                    self._process_objective_functions(network_instance)
                    # self._process_task_characterization(network_instance)

    def __addclasses(self) -> None:

        new_classes = {
            "Self-Supervised Classification": self.ontology.TaskCharacterization,
            "Unsupervised Classification": self.ontology.TaskCharacterization,
        }

        for name, parent in new_classes.items():
            try:
                create_subclass(self.ontology, name, parent)
            except Exception as e:
                self.logger.error(f"Error creating new class {name}: {e}")

    def run(self) -> None:
        """
        Main method to run the ontology instantiation process.
        """
        try:
            if not hasattr(self.ontology, "ANNConfiguration"):
                raise AttributeError("Error: Class 'ANNConfiguration' not found in ontology.")

            # Initialize the LLM engine with the document context.
            init_engine(self.ann_config_name, self.json_file_path)

            self.__addclasses()  # Add new classes to ontology

            # Could grab model name from user input or JSON file.
            # Extract the title from the JSON file.
            # try:
            #     with open(self.json_file_path, 'r', encoding='utf-8') as file:
            #         data = load(file)
            #     titles = [item['metadata']['title'] for item in data if 'metadata' in item and 'title' in item['metadata']]
            #     title = titles[0] if titles else "DefaultTitle"
            # except Exception as e:
            #     self.logger.error(f"Error reading JSON file: {e}")
            #     title = "DefaultTitle"

            ann_config_instance = self._instantiate_cls(
                self.ontology.ANNConfiguration, self.ann_config_name
            )

            # Process the network class and it's components.
            self._process_network(ann_config_instance)

            # Process TrainingStrategy and it's components.
            # self._process_training_strategy(ann_config_instance)

            # extract information about the application of the ANN (i.e. cancer detection)
            # what kind of data, image, text
            # Shapes of layers

            self.logger.info("An ANN has been successfully instantiated.")

        except Exception as e:
            self.logger.error(f"Error during ontology instantiation: {e}")
            raise e


if __name__ == "__main__":
    import time

    start_time = time.time()
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()

    # LLM based stuff
    # for model_name in ["alexnet", "resnet", "vgg16", "gan"]:
    #     with ontology:
    #         try:
    #             instantiator = OntologyInstantiator(
    #                 ontology, f"data/{model_name}/doc_{model_name}.json", model_name
    #             )
    #             instantiator.run()
    #         except Exception as e:
    #             logging.error(f"Error instantiating the {model_name} ontology: {e}")
    #             continue

    # new_file_path = "tests/papers_rag/annett-o-test(2).owl"  # Assume test file for now.
    # ontology.save(file=new_file_path, format="rdfxml")
    # logging.info(f"Ontology saved to {new_file_path}")

    # code for graphdb extraction & instantation
    inst = OntologyInstantiator(ontology , f"data/alexnet/alexnet_code0.json")
    inst.process_layers(network_instance=0)

    # Move saving outside the loop to ensure all networks are stored in the final ontology
    new_file_path = "tests/papers_rag/annett-o-test(2).owl"  # Assume test file for now.
    ontology.save(file=new_file_path, format="rdfxml")
    logging.info(f"Ontology saved to {new_file_path}")

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    logging.info(f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds.")
    logging.info("Ontology instantiation completed.")
