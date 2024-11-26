from owlready2 import *
from typing import List, Set
import warnings

""" 1) Class Functions """

def get_class_properties(ontology: Ontology, onto_class: ThingClass) -> List[Property]:
    """
    Retrieves all properties in the given ontology that have the specified class as their domain.

    Args:
        ontology (Ontology): The ontology containing the properties.
        onto_class (ThingClass): The class for which to find properties with this domain.

    Returns:
        List[Property]: A list of properties that have the specified class as their domain.
    """
    return [prop for prop in ontology.properties() if onto_class in prop.domain]



def get_property_range_type(property: Property) -> str:
    """
    Determines if the range of a property is atomic or refers to another class.

    Args:
        property (Property): The property for which to check the range.

    Returns:
        str: "atomic" if the range is a primitive datatype, otherwise "class".
    """
    
    if not property.range:
        return "atomic"

    for range_type in property.range:
        # Check if the range refers to a class (ThingClass)
        if isinstance(range_type, ThingClass):
            return "class"
        # Check if the range is a known atomic data type
        if range_type in [str, int, float, bool]:
            return "atomic"

    # default to atomic
    return "atomic"

def get_class_restrictions(onto_class: ThingClass) -> List[Restriction]:
    """
    Retrieves all restrictions (including cardinality restrictions) applied to a specified class.

    Args:
        onto_class (ThingClass): The class for which to retrieve restrictions.

    Returns:
        List[Restriction]: A list of restrictions applied to the class.
    """
    return [restriction for restriction in onto_class.is_a if isinstance(restriction, Restriction)]

# def get_all_subclasses(onto_class: ThingClass) -> List[ThingClass]:
#     """
#     Retrieves all direct and indirect subclasses of a specified class.

#     Args:
#         onto_class (ThingClass): The class for which to find subclasses.

#     Returns:
#         List[ThingClass]: A list of all subclasses of the specified class.
#     """
#     return list(onto_class.subclasses())

def create_class(ontology: Ontology, class_name: str, base_class: ThingClass = None) -> ThingClass:
    """
    Dynamically creates a class in the ontology if it does not already exist.

    Args:
        ontology (Ontology): The ontology in which to create the class.
        class_name (str): The name of the class to create.
        base_class (ThingClass, optional): The base class for the new class. Defaults to None, which uses Thing.

    Returns:
        ThingClass: The newly created class or the existing class if it already exists.
    """
    # Check if the class already exists
    existing_class = getattr(ontology, class_name) #has_attr behavior working incorrectly
    if existing_class is not None:
        warnings.warn(f"Class '{class_name}' already exists.")
        return existing_class

    # Set base class to Thing if no base_class is provided
    if base_class is None:
        base_class = ontology.Thing 

    # Dynamically create the new class using `type()`
    new_class = type(class_name, (base_class,), {"namespace": ontology})
    setattr(ontology, class_name, new_class)  # Add the new class to the ontology's namespace
    # print(f"Class '{class_name}' created with base '{base_class.__name__}'.")
    return new_class

def create_subclass(ontology: Ontology, class_name: str, base_class: ThingClass) -> ThingClass:
    """
    Dynamically creates a subclass in the ontology if it does not already exist.

    Args:
        ontology (Ontology): The ontology in which to create the class.
        class_name (str): The name of the class to create.
        base_class (ThingClass): The base class for the new class.

    Returns:
        ThingClass: The newly created class or the existing class if it already exists.
    """
    # Enforce base_class
    if base_class is None:
        warnings.warn(f"No base class defined for subclass '{class_name}'.")
        return None
    # Create class with base_class
    return create_class(ontology=ontology, class_name=class_name, base_class=base_class)

""" 2) Instance Functions """

def get_instance_class_properties(ontology: Ontology, instance: Thing) -> List[Property]:
    """
    Retrieves all properties in the ontology that are applicable to the class of a given instance.

    Args:
        ontology (Ontology): The ontology containing the properties.
        instance (Thing): The instance for which to retrieve class-based properties.

    Returns:
        List[Property]: A list of properties that are applicable to the instance's class.
    """
    # Get the primary class of the instance (assuming single inheritance)
    instance_class = instance.is_a[0] if instance.is_a else None
    if not instance_class:
        return []
    
    # Retrieve properties based on the instance's class domain
    return get_class_properties(ontology, instance_class)

def get_instantiated_properties(instance: Thing) -> List[Property]:
    """
    Retrieves all properties that have been set (instantiated) on the specified instance.

    Args:
        instance (Thing): The instance for which to retrieve instantiated properties.

    Returns:
        List[Property]: A list of properties that have been set on the instance.
    """
    instantiated_properties = []
    for prop in instance.get_properties():
        if instance.__getattr__(prop.name):  # Check if the property has a non-empty value
            instantiated_properties.append(prop)
    return instantiated_properties

def get_instantiated_property_values(instance: Thing) -> dict:
    """
    Retrieves a dictionary of instantiated properties and their values for the specified instance.

    Args:
        instance (Thing): The instance for which to retrieve property values.

    Returns:
        dict: A dictionary where keys are property names and values are the values set for each property.
    """
    property_values = {}
    for prop in get_instantiated_properties(instance):
        property_values[prop.name] = instance.__getattr__(prop.name)
    return property_values

""" 3) Property Functions """

# def get_property_restrictions()


"""Richie written"""

def get_base_class(onto:Ontology):
    return onto.ANNConfiguration

def get_object_properties_for_class(ontology: Ontology,cls: ThingClass):
    """
    Retrieves all object properties where the given class is in the domain.

    Args:
        cls (ThingClass): The class for which to find object properties.
        ontology (Ontology): The ontology containing the properties.

    Returns:
        Set[ObjectPropertyClass]: A set of object properties with the class in their domain.
    """
    return {prop for prop in ontology.object_properties() if cls in prop.domain}
    # return {prop for prop in ontology.properties() if cls in prop.domain}



def print_instantiated_classes_and_properties(ontology: Ontology):
    """
    Prints all instantiated classes and their properties in the given ontology.

    Args:
        ontology (Ontology): The ontology to inspect.

    """
    print("Instantiated Classes and Properties:")
    for instance in ontology.individuals():
        print(f"Instance: {instance.name}")
        # Get the classes this instance belongs to
        classes = instance.is_a
        class_names = [cls.name for cls in classes if cls.name]
        print(f"  Classes: {', '.join(class_names) if class_names else 'None'}")
        
        # Get instantiated properties and their values
        properties = instance.get_properties()
        for prop in properties:
            values = instance.__getattr__(prop.name)
            if values:  # Only print properties that have values
                if isinstance(values, list):
                    values_str = ", ".join(str(v) for v in values)
                else:
                    values_str = str(values)
                print(f"  Property: {prop.name}, Values: {values_str}")
        print("-" * 40)

# Extra functions i wrote
def list_owl_classes(onto: Ontology):
    # List all classes
    print("Classes in the ontology:")
    for cls in onto.classes():
        print(f"- {cls.name}")

def list_owl_object_properties(onto: Ontology):
    # List all object properties
    print("\nObject Properties in the ontology:")
    for prop in onto.object_properties():
        print(f"- {prop.name}")

def list_owl_data_properties(onto: Ontology):
    # List all data properties
    print("\nData Properties in the ontology:")
    for prop in onto.data_properties():
        print(f"- {prop.name}")

def get_property_class_range(properties):
    """
    Retrieves the range (classes) of given properties.

    :param properties: List of properties whose ranges are to be found.
    :return: Dictionary mapping property names to lists of class names in their range.
    """
    connected_classes = {}
    for prop in properties:
        try:
            connected_classes[prop.name] = [
                cls.name for cls in prop.range if hasattr(cls, 'name')
            ]
        except AttributeError as e:
            print(f"Error processing property {prop.name}: {e}")
            connected_classes[prop.name] = []
    return connected_classes


def iterate_subclasses(cls: ThingClass, level: int=0, visited_classes: Set[ThingClass]=None, onto: Ontology=None):
    """
    Recursively iterates through subclasses and traverses object property ranges.
    
    Args:
        cls (ThingClass): The class to start the traversal.
        level (int): The current recursion level.
        visited_classes (Set[ThingClass]): Tracks visited classes to avoid infinite loops.
        file: A file object (unused in the current function).
        ontology (Ontology): The ontology being traversed.
    
    Yields:
        ThingClass: Each class in the traversal.

    Usage: 
        for cls in iterate_subclasses(root_class,onto=self.ontology):
            print(f"Class: {cls.name}") # Example call
    """
    if cls is None:
        raise ValueError("Parameter 'cls' must be provided and cannot be None.")
    if not isinstance(onto, Ontology):
        raise ValueError("Parameter 'ontology' must be an instance of 'Ontology'.")


    if visited_classes is None:
        visited_classes = set()
    if cls in visited_classes:
        return
    visited_classes.add(cls)

    # Get data properties and object properties
    obj_props = get_object_properties_for_class(onto,cls)
    yield cls
    # Write object properties to file and recursively traverse range classes
    if obj_props:
        for prop in obj_props:
            for range_cls in prop.range:
                if isinstance(range_cls, ThingClass):
                    yield from iterate_subclasses(range_cls, level + 1, visited_classes, onto)