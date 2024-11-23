# Load your ontology (replace 'your_ontology.owl' with your actual file)
from owlready2 import *
from utils.constants import Constants as C

onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
# from typing import List, Dict, Union, Optional
# import types

from owlready2 import Ontology, ThingClass, DataPropertyClass, ObjectPropertyClass, Thing
from typing import Set

from owlready2 import Ontology, ThingClass, DataPropertyClass, ObjectPropertyClass
from typing import Set

def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from ANNConfiguration, including data properties and object properties,
    to a file. It recursively traverses object property ranges as a tree.

    Args:
        ontology (Ontology): The ontology to document.
        file_path (str): The path to the file where the structure will be written.
    """
    with open(file_path, 'w') as file:
        # Start from ANNConfiguration
        if hasattr(ontology, 'ANNConfiguration'):
            visited_classes = set()
            write_class_structure_to_file(ontology.ANNConfiguration, 0, visited_classes, file, ontology)
        else:
            print("Class 'ANNConfiguration' does not exist in the ontology.")

def write_class_structure_to_file(cls: ThingClass, level: int, visited_classes: Set[ThingClass], file, ontology: Ontology):
    """
    Writes the structure of a class to a file, including its data properties and object properties.
    Recursively traverses classes in the range of object properties.

    Args:
        cls (ThingClass): The class to document.
        level (int): The current level in the hierarchy for indentation.
        visited_classes (Set[ThingClass]): Set of already visited classes to prevent infinite loops.
        file: The file object to write to.
        ontology (Ontology): The ontology containing the classes and properties.
    """
    indent = '  ' * level
    if cls in visited_classes:
        return
    visited_classes.add(cls)
    file.write(f"{indent}- Class: {cls.name}\n")

    # Get data properties and object properties
    data_props = get_data_properties_for_class(cls, ontology)
    obj_props = get_object_properties_for_class(cls, ontology)

    # Write data properties to file
    if data_props:
        file.write(f"{indent}  Data Properties:\n")
        for prop in data_props:
            file.write(f"{indent}    - {prop.name}\n")

    # Write object properties to file and recursively traverse range classes
    if obj_props:
        file.write(f"{indent}  Object Properties:\n")
        for prop in obj_props:
            file.write(f"{indent}    - {prop.name}\n")
            for range_cls in prop.range:
                if isinstance(range_cls, ThingClass):
                    write_class_structure_to_file(range_cls, level + 1, visited_classes, file, ontology)

def get_data_properties_for_class(cls: ThingClass, ontology: Ontology) -> Set[DataPropertyClass]:
    """
    Retrieves all data properties where the given class is in the domain.

    Args:
        cls (ThingClass): The class for which to find data properties.
        ontology (Ontology): The ontology containing the properties.

    Returns:
        Set[DataPropertyClass]: A set of data properties with the class in their domain.
    """
    return {prop for prop in ontology.data_properties() if cls in prop.domain}

def get_object_properties_for_class(cls: ThingClass, ontology: Ontology) -> Set[ObjectPropertyClass]:
    """
    Retrieves all object properties where the given class is in the domain.

    Args:
        cls (ThingClass): The class for which to find object properties.
        ontology (Ontology): The ontology containing the properties.

    Returns:
        Set[ObjectPropertyClass]: A set of object properties with the class in their domain.
    """
    return {prop for prop in ontology.object_properties() if cls in prop.domain}


def print_class_structure(onto_class: ThingClass, level=0, visited=None):
    """
    Prints the structure of a specified class, including its subclasses, data properties, and object properties.
    
    Args:
        onto_class (ThingClass): The class whose structure is to be printed.
        level (int, optional): The current level in the hierarchy for indentation.
        visited (set, optional): Set of visited classes to prevent infinite loops.
    """
    if visited is None:
        visited = set()
    indent = '  ' * level
    if onto_class in visited:
        return
    visited.add(onto_class)
    print(f"{indent}- Class: {onto_class.name}")
    
    # Data properties
    data_props = [prop for prop in onto_class.get_class_properties() if isinstance(prop, DataPropertyClass)]
    if data_props:
        print(f"{indent}  Data Properties:")
        for prop in data_props:
            print(f"{indent}    - {prop.name}")
    
    # Object properties
    obj_props = [prop for prop in onto_class.get_class_properties() if isinstance(prop, ObjectPropertyClass)]
    if obj_props:
        print(f"{indent}  Object Properties:")
        for prop in obj_props:
            print(f"{indent}    - {prop.name}")
    
    # Subclasses
    subclasses = onto_class.subclasses()
    for subclass in subclasses:
        print_class_structure(subclass, level + 1, visited)

def print_instance_structure(instance: Thing, level=0, visited=None):
    """
    Prints the structure for a given root of an instantiated class, showing properties and their values.
    
    Args:
        instance (Thing): The instance to inspect.
        level (int, optional): The current level in the hierarchy for indentation.
        visited (set, optional): Set of visited instances to prevent infinite loops.
    """
    if visited is None:
        visited = set()
    indent = '  ' * level
    if instance in visited:
        return
    visited.add(instance)
    print(f"{indent}- Instance: {instance.name}")
    class_names = [cls.name for cls in instance.is_a if cls.name]
    print(f"{indent}  Classes: {', '.join(class_names)}")
    
    # Properties and their values
    for prop in instance.get_properties():
        values = prop[instance]
        if values:
            if isinstance(prop, ObjectPropertyClass):
                print(f"{indent}  Object Property: {prop.name}")
                for value in values:
                    print_instance_structure(value, level + 1, visited)
            elif isinstance(prop, DataPropertyClass):
                values_str = ', '.join(str(v) for v in values)
                print(f"{indent}  Data Property: {prop.name}, Value: {values_str}")

def print_instances_of_class(onto_class: ThingClass):
    """
    Prints all instances of a given class.
    
    Args:
        onto_class (ThingClass): The class whose instances are to be printed.

    Example input:
        onto.TaskCharacterization
    """
    instances = list(onto_class.instances())
    if instances:
        print(f"Instances of class '{onto_class.name}':")
        for instance in instances:
            print(f"- {instance.name}")
    else:
        print(f"No instances found for class '{onto_class.name}'.")


def print_instances_of_data_property(data_property: DataPropertyClass):
    """
    Prints all instances (values) of a given data property.

    Args:
        data_property (DataPropertyClass): The data property whose instances (values) are to be printed.

    Example input:
        onto.layer_num_units
    """
    instances = data_property.get_relations()
    if instances:
        print(f"Values for data property '{data_property.name}':")
        for subject, value in instances:
            print(f"- Instance: {subject.name} | Value: {value}")
    else:
        print(f"No values found for data property '{data_property.name}'.")



# print_instances_of_class(onto.layer_num_units)
print_instances_of_data_property(onto.layer_num_units)

# print_instances_of_class(onto.TaskCharacterization)


# Write ontology structure to a file
# write_ontology_structure_to_file(onto, './ontology_structure.txt')

# Print class structure starting from a specific class
# print_class_structure(onto.GAN)

# Print instance structure
# john = onto.Person("john_doe")
# john.hasName = "John Doe"
# john.hasAge = 30

# print_instance_structure(john)
