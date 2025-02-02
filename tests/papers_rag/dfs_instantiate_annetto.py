from owlready2 import Ontology, ThingClass, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties, get_connected_classes, get_subclasses
)
from utils.annetto_utils import requires_final_instantiation, subclasses_requires_final_instantiation

OMIT_CLASSES = ["DataCharacterization", "Regularization"]

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


    def _write_data_properties(indent: str, cls: ThingClass):
        """Write all data properties of a class, if any."""
        props = get_class_data_properties(ontology, cls)
        if props:
            for prop in props:
                # file.write(f"{indent}        - Data Prop: {prop.name} (atomic)\n")
                pass

    def _instantiate_cls(cls:ThingClass, instance_names:str):
        pass

    def _query_llm(instructions, prompt):
        pass
    
    def _get_cls_prompt(cls:ThingClass):
        pass

    def _get_cls_instances(cls:ThingClass):
        # Get prompt for given class

        # combine prompt with RAG context

        # combine prompt with general llm instructions 

        # Query LLM on prompt

        # Validate llm response format (i.e. pydantic for json)

        # Parse prompt into list of instance names

        # list of instance objects = _instantiate_cls (instance_names)

        # return list of instance objects
        return ["Convolutional Layer", "Fully-Connected Layer", "Attention Layer"]

    

    def _process_entity(cls: ThingClass, label: str, processed_classes: set, ancestors: list[str] = None):
        """
        Process an entity (class, connected class, or subclass)
        """

        if ancestors is None:
            ancestors = []

        # Skip if already processed or omitted.
        if cls in processed_classes or cls.name in OMIT_CLASSES:
            return

        processed_classes.add(cls)

        # Process for list of instance objects
        instances = _get_cls_instances(cls)

        if not instances:
            print(f"No instances for {cls}")
            return

        for instance_name in instances:

            # Append current instance to ancestor chain.
            new_ancestors = ancestors + [instance_name]
            
            # Maybe here instantiate the instance, record it, or pass it as context
            # to generate further prompts.
            _instantiate_cls(cls, new_ancestors) 
            
            # If your instances need further processing (e.g., they have subclasses or connected classes),
            # you can get the corresponding ThingClass for the instance and call _process_entity on it.
            # For example:
            instance_cls = get_corresponding_class(instance_name, ontology)  # You may need to implement this.
            if instance_cls is not None:
                _process_entity(instance_cls, "Instance", processed_classes, new_ancestors)



            # TODO: somehow the recurison needs to be split so for each instance 

            # Process data properties into instance object (probably?).

            # Process connected classes via object properties.
            connected = get_connected_classes(cls, ontology)
            if connected:
                for conn in connected:
                    if isinstance(conn, ThingClass):
                        _process_entity(conn, "Connected Class", processed_classes)
                    else:
                        print(f"Non-Class Connection? ################\n")

            # Process subclasses.
            subclasses = get_subclasses(cls)
            if subclasses:
                for subclass in subclasses:
                    _process_entity(subclass, "Subclass", processed_classes)

    # Check for the required key class.
    if not hasattr(ontology, 'ANNConfiguration'):
        print("Error: Class 'ANNConfiguration' not found in ontology.")
        return

    processed_classes = set()

    # Process the top-level classes.
    if hasattr(ontology, "Network"):
        _process_entity(ontology.Network, "Class", processed_classes)
    if hasattr(ontology, "TrainingStrategy"):
        _process_entity(ontology.TrainingStrategy, "Class", processed_classes)

    print("An ANN has been instantiated.")

if __name__ == "__main__":
    OUTPUT_FILE = './test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()    

    dfs_instantiate_annetto(ontology)