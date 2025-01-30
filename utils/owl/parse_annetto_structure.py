from owlready2 import *
from utils.owl.owl import *
from utils.constants import Constants as C


def requires_final_instantiation(cls, ontology):
    """
    Determines if a class requires final instantiation as a standalone object.

    A class requires final instantiation if it:
        - Has no data properties and has no object properties.
        or
        - Has no subclasses

    Args:
        cls (ThingClass): the class to check.
        ontology (Ontology): the ontology to which the class belongs.

    Returns:
        bool: True if the class must be instantiated as a final object, otherwise False.
    """
    has_data_properties = any(cls in prop.domain for prop in ontology.data_properties())
    has_object_properties = any(cls in prop.domain for prop in ontology.object_properties())
    return (not has_data_properties and not has_object_properties) or not get_subclasses(cls)