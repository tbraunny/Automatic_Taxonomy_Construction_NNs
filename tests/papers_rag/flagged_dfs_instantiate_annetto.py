from owlready2 import Ontology, ThingClass, Thing, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties, get_connected_classes, get_subclasses, 
    create_cls_instance
)
from utils.annetto_utils import requires_final_instantiation, subclasses_requires_final_instantiation
import re

from utils.llm_service import init_engine, query_llm

OMIT_CLASSES = ["DataCharacterization", "Regularization"]

PARENT_CLASSES = set(["LossFunction", "RegularizerFunction", "ActivationLayer", "NonDiff", "Smooth", "AggregationLayer", "NoiseLayer"])

# from pydantic import BaseModel, ValidationError
# from typing import List
# class LLMResponse(BaseModel):
#     """
#     Data model for validating LLM responses.
#     Attributes:
#         instance_names (List[str]): List of ontology class names extracted from the LLM's response.
#     """
#     instance_names: List[str]

# def validate_response(response_json:str):
#         """Validates and parses the JSON response."""
#         response_json = response_json if isinstance(response_json, dict) else json.loads(response.strip())
#         return LLMResponse.model_validate(response_json)

# def extract_JSON(response: str) -> dict:
#     """
#     Extracts JSON data from a response string.

#     Args:
#         response (str): The LLM's response containing JSON data.

#     Returns:
#         dict: Extracted JSON object.

#     Raises:
#         ValueError: If no valid JSON block is found in the response.
#     """
#     try:
#         json_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
#         if json_match:
#             return json.loads(json_match.group(1))
#         raise ValueError("No valid JSON block found in the response.")
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Error decoding JSON: {e}\nResponse: {response}")

def dfs_instantiate_annetto(ontology: Ontology):
    """
    
    """
    def _needs_instantiation(cls: ThingClass) -> bool:
        """Checks if a class should be instantiated."""
        return requires_final_instantiation(cls)
    
    def _needs_subclass_instantiation(cls: ThingClass) -> bool:
        """Checks if new subclasses should be explored."""
        return subclasses_requires_final_instantiation(cls)


    def _instantiate_data_property(cls:ThingClass, instance:Thing, ancestor_things: list[Thing]=None):
        """Write all data properties of a class, if any."""
        props = get_class_data_properties(ontology, cls)
        if props:
            for prop in props:
                # file.write(f"{indent}        - Data Prop: {prop.name} (atomic)\n")

                # Query llm if a value for this data property exists

                # Add it to instance
                pass

    def _instantiate_cls(cls:ThingClass, instance_names:str) -> Thing:
        pass

    def _query_llm(instructions, prompt)-> str:
        return ["Convolutional Network 1", "Fully Con 3"]
    
    def _get_cls_prompt(cls:ThingClass) -> str:
        """Given a class, get an associated llm prompt for it to be instantiated"""
        return "What kind of neural networks are this model? (i.e. A typical GAN model will have a generator network and a discriminator network.)"

    def _get_cls_instances(cls:ThingClass) -> Thing:
        # Get prompt for given class

        # combine prompt with RAG context

        # combine prompt with general llm instructions 

        # Query LLM on prompt

        # Validate llm response format (i.e. pydantic for json)

        # Parse prompt into list of instance names

        # list of instance objects = _instantiate_cls (instance_names)

        # return list of instance objects
        return ["Convolutional Layer", "Fully-Connected Layer", "Attention Layer"]
    
    def get_cls_definition(cls):
        return """An Activation Layer in a neural network applies an activation function to the input data, introducing non-linearity to the model, which enables the network to learn complex patterns. It transforms the weighted sum of inputs in a layer before passing it to the next layer."""
    
    def split_camel_case(names:list) -> list:
        if isinstance(names, str):  # If a single string is passed, convert it into a list
            names = [names]

        split_names = []
        for name in names:
            if re.fullmatch(r'[A-Z]{2,}[a-z]*$', name):  # Skip all-uppercase acronyms like "RNRtop"
                split_names.append(name)
            else:
                # Split between lowercase-uppercase (e.g., "NoCoffee" → "No Coffee")
                name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)

                # Split when a sequence of uppercase letters is followed by a lowercase letter
                # (e.g., "CNNModel" → "CNN Model")
                name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)

                split_names.append(name)

        return split_names
        
    def _get_subclasses_instances(cls:ThingClass) -> Thing:
        # Get prompt for given class

        ### Assumptions ###
        network_name = "ConvolutionalNetwork"

        # Gets list of subclasses
        subclasses = get_subclasses(cls)
        subclass_names = [subclass.name for subclass in subclasses]

        print(subclass_names)

        class_name = split_camel_case(cls.name)
        network_name = split_camel_case(network_name)

        print(class_name, network_name,subclass_names)

        class_definition = get_cls_definition(cls)

        prompt = (
            f"""Name each instance of {class_name} in the {network_name}."""
            f"""The definition of {class_name} is {class_definition}.\n"""
            f"""Examples and the appropriate output format are: {subclass_names}.\n"""
        )

        # from random import sample
        # random_numbers = sample(range(1, 16), 3)
        # formatted_names = ", ".join(f"{name}_{num}" for name, num in zip(subclass_names, random_numbers)) + ""

        # Query LLM on prompt
        # named_instances = query_llm(prompt)

        print("Named Instances", named_instances)

        # Parse prompt into list of instance names

        # list of instance objects = _instantiate_cls (instance_names)

        # return list of instance objects
        return ["Convolutional Layer", "Fully-Connected Layer", "Attention Layer"]

    

    def _process_entity(cls: ThingClass, label: str, processed_classes: set, ancestor_things: list[Thing] = None):
        """
        Process an entity (class, connected class, or subclass)
        """

        cls = ontology.ActivationLayer

        if ancestor_things is None:
            ancestor_things = []

        # Skip if already processed, preventing loops
        # Skip if in omit list
        if cls in processed_classes or cls.name in OMIT_CLASSES:
            return

        processed_classes.add(cls)

        if cls.name in PARENT_CLASSES:
            # Process parent class by checking if any of its subclasses exists, and if not creating new class blah 
            # Process parent class; Check if any
            instances = _get_subclasses_instances(cls)

        if not instances:
            print(f"No instances for {cls}")
            return

        for instance in instances:

            # Append current instance to ancestor chain.
            new_ancestor_things = ancestor_things + [instance]
            
            # Maybe here instantiate the instance, record it, or pass it as context
            # to generate further prompts.
            _instantiate_cls(cls, new_ancestor_things) 

            # TODO: somehow the recurison needs to be split so for each instance 

            # Process data properties into instance object (probably?).
            _instantiate_data_property(cls, instance, new_ancestor_things)

            # Process connected classes via object properties.
            connected = get_connected_classes(cls, ontology)
            if connected:
                for conn in connected:
                    if isinstance(conn, ThingClass):
                        _process_entity(conn, "Connected Class", processed_classes, ancestor_things)
                    else:
                        print(f"Non-Class Connection? ################\n")

            # Process subclasses.
            subclasses = get_subclasses(cls)
            if subclasses:
                for subclass in subclasses:
                    _process_entity(subclass, "Subclass", processed_classes, ancestor_things)

    # Check for the required key class.
    if not hasattr(ontology, 'ANNConfiguration'):
        print("Error: Class 'ANNConfiguration' not found in ontology.")
        return
    
    json_file_path = "data/alexnet/doc_alexnet.json"
    init_engine(json_file_path)

    processed_classes = set()

    root_instance = create_cls_instance(ontology.ANNConfiguration, "ImageNet Classification with Deep Convolutional Neural Networks")

    # Process the top-level classes.
    if hasattr(ontology, "Network"):
        _process_entity(ontology.Network, "Class", processed_classes)
    # if hasattr(ontology, "TrainingStrategy"):
    #     _process_entity(ontology.TrainingStrategy, "Class", processed_classes)

    # Need to have instances of Network be in the range of object hasNetwork and root_instance be in the domain
    # Need to have instances of TrainingStrategy be in the range of object hasTrainingStrategy and root_instance be in the domain


    print("An ANN has been instantiated.")

if __name__ == "__main__":
    OUTPUT_FILE = './test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()    

    dfs_instantiate_annetto(ontology)