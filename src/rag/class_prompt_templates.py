from owlready2 import *
from typing import List, Dict, Optional
from utils.constants import Constants as C


def get_possible_subclasses(cls: ThingClass, visited=None) -> List[str]:
    """
    Recursively determines all possible subclasses of a given OWL class,
    while avoiding cycles in subclass relationships.

    Args:
        cls (ThingClass): The OWL class to explore for subclasses.
        visited (set): A set of visited class names to avoid cycles.

    Returns:
        list: A list of subclass names.
    """
    if visited is None:
        visited = set()

    if cls.name in visited:
        return []  # Avoid cycles

    visited.add(cls.name)
    subclasses = []

    for subclass in cls.subclasses():
        subclasses.append(subclass.name)
        subclasses.extend(get_possible_subclasses(subclass, visited))  # Recursive call

    return list(set(subclasses))  # Remove duplicates


def get_class_prompt(
    cls: ThingClass, 
    consider_subclasses: bool = False, 
    parent_instance: Optional[str] = None
) -> Dict[str, str]:
    """
    Generates a question for a given OWL class to determine the instances to be created.

    Args:
        cls (ThingClass): The OWL class to generate the prompt for.
        consider_subclasses (bool): Whether to include subclasses in the prompt.
        parent_instance (str, optional): The name of the parent instance to provide context.

    Returns:
        dict: A dictionary with the class name and the corresponding question.
    """
    if not isinstance(cls, ThingClass):
        raise ValueError("Input must be of type ThingClass.")
    
    # Class name
    cls_name = cls.name.lower()

    # Subclass names if applicable
    subclass_context = ""
    if consider_subclasses:
        possible_subclasses = get_possible_subclasses(cls)
        if possible_subclasses:
            subclass_context = f" Include the following subclasses in your response: {', '.join(possible_subclasses)}."

    # Parent context
    parent_context = (
        f" based on the given context associated with the parent node '{parent_instance}'"
        if parent_instance
        else ""
    )

    question = (
        f"Identify all {cls_name}s based on the paper and{parent_context}. "
        f"For each identified {cls_name}, assign a unique name in the format '<configuration_name>_{cls_name}X'."
        f"{subclass_context}"
    )

    return {
        "class_name": cls_name,
        "question": question
    }

# example usage
if __name__ == "__main__":
    # Load ontology using owlready2
    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

    with onto:
        class Network(Thing):
            pass

        class Layer(Thing):
            pass

        class ObjectiveFunction(Thing):
            pass

        class ActivationFunction(Thing):
            pass

        class Optimizer(Thing):
            pass

        class Dataset(Thing):
            pass

    print(get_class_prompt(Network, consider_subclasses=True, parent_instance="ANNConfig1"))
    print('\n')
    print(get_class_prompt(Layer, consider_subclasses=True, parent_instance="Network1"))
    print('\n')
    print(get_class_prompt(ObjectiveFunction, consider_subclasses=False, parent_instance="Network1"))
    print('\n')
    print(get_class_prompt(ActivationFunction, consider_subclasses=False, parent_instance="Layer1"))
    print('\n')
    print(get_class_prompt(Optimizer, consider_subclasses=False, parent_instance="ANNConfig1"))
    print('\n')
    print(get_class_prompt(Dataset, consider_subclasses=False))
