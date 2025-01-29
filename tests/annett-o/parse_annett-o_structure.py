from owlready2 import *
from typing import List, Dict

# # Load ontology

onto = get_ontology("./ontology/annett-o-0.1.owl").load()

class ConvolutionalNeuralNetwork(onto.ANNConfiguration):
    pass

# print(onto.ConvolutionalNeuralNetwork.is_a)

AlexNet = ConvolutionalNeuralNetwork("Alex_Net")

# Network = onto.Network

AlexNet_Network = onto.Network("AlexNet_Network")

AlexNet.hasNetwork = [AlexNet_Network]
                                     
print(AlexNet.hasNetwork)

# ann_config_instances = onto.ANNConfiguration.instances()
# print(ann_config_instances)

# Access all instances of ANNConfiguration directly as Python objects
# ann_config_instances = onto.ANNConfiguration.instances()

# for ann_config in ann_config_instances:
#     networks = ann_config.hasNetwork  # access related networks directly as attributes
#     print(f"Networks for {ann_config.name}: {[net.name for net in networks]}")


def explore_properties(cls, level=0, visited_classes=None):
    """
    Recursively explore all properties (ObjectProperties and DataProperties) of a class and its related classes.

    Args:
        cls: The OWL class to start exploring from.
        level: The current depth level for formatting the output.
        visited_classes: A set of already visited classes to avoid infinite recursion.
    """
    # Initialize visited_classes on the first call
    if visited_classes is None:
        visited_classes = set()
    
    # Avoid revisiting classes to prevent infinite recursion
    if cls in visited_classes:
        return
    
    # Mark the current class as visited
    visited_classes.add(cls)

    indent = "  " * level  # Indentation for visual hierarchy
    print(f"{indent}Class: {cls.name}")
    
    # Explore all properties (both ObjectProperties and DataProperties)
    for prop in cls.get_class_properties():
        if isinstance(prop, ObjectPropertyClass):
            # ObjectProperty pointing to other classes
            range_classes = prop.range
            print(f"{indent}  Relationship (ObjectProperty): {prop.name}")
            print(f"{indent}    Range Class(es): {[rc.name for rc in range_classes]}")

            # Recursively explore each class in the range of this ObjectProperty
            for range_class in range_classes:
                explore_properties(range_class, level + 2, visited_classes)
        
        elif isinstance(prop, DataPropertyClass):
            # DataProperty with literal values
            print(f"{indent}  Property (DataProperty): {prop.name}")
            print(f"{indent}    Data Range: {prop.range}")

        else:
            # Handle any other property types if they exist (for completeness)
            print(f"{indent}  Other Property: {prop.name}")
    
    # Explore any additional properties directly associated with instances
    print(f"{indent}Instance Properties:")
    for instance in cls.instances():
        for inst_prop in instance.get_properties():
            inst_values = getattr(instance, inst_prop.name, None)  # Get values safely
            print(f"{indent}  Instance {instance.name} - Property {inst_prop.name}: {inst_values}")


def explore_context(entity, level=0, visited=None) -> List[Dict[str, any]]:
    """
    Recursively explores the properties and classes associated with a given entity.

    Args:
        entity: The starting class or instance in the ontology.
        level: The current depth level in the recursive exploration.
        visited: A set of classes that have already been visited to avoid circular references.

    Returns:
        A list of dictionaries representing the class, properties, and possible range classes.
    """
    if visited is None:
        visited = set()

    indent = " " * level * 2
    context_info = []

    # Avoid revisiting classes to prevent infinite recursion
    if entity in visited:
        return []
    visited.add(entity)

    # Gather object and data properties
    for prop in entity.get_class_properties():
        if isinstance(prop, ObjectPropertyClass):
            range_classes = prop.range
            property_info = {
                "level": level,
                "property_name": prop.name,
                "type": "ObjectProperty",
                "range_classes": [cls.name for cls in range_classes]
            }
            print(f"{indent}ObjectProperty: {prop.name} - Range: {property_info['range_classes']}")

            # Recurse into each range class to explore properties further down
            for range_class in range_classes:
                property_info["nested"] = explore_context(range_class, level + 1, visited)

            context_info.append(property_info)
        
        elif isinstance(prop, DataPropertyClass):
            data_ranges = prop.range
            property_info = {
                "level": level,
                "property_name": prop.name,
                "type": "DataProperty",
                "data_range": [str(data_range) for data_range in data_ranges]
            }
            print(f"{indent}DataProperty: {prop.name} - Data Range: {property_info['data_range']}")
            context_info.append(property_info)

    return context_info

# # Example of using the function on ANNConfiguration
ann_config_class = onto.ANNConfiguration
context_info = explore_context(ann_config_class)