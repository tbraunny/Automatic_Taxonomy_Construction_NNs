from owlready2 import Ontology, ThingClass, Property, Restriction, Thing
from typing import List
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