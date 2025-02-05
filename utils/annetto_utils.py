from owlready2 import ThingClass

def requires_final_instantiation(cls: ThingClass) -> bool:
        """
        Determines if a class requires instantiation as a standalone object.

        A class requires instantiation if has no subclasses.

        :param cls: The class to check.
        :return: True if the class requires final instantiation, else False.
        """
        # has_data_properties = any(cls in prop.domain for prop in ontology.data_properties())
        # has_object_properties = any(cls in prop.domain for prop in ontology.object_properties())
        has_subclasses = any(list(cls.subclasses()))
        return not has_subclasses

def subclasses_requires_final_instantiation(cls: ThingClass) -> bool:
    """
    Determines if a class is a direct parent of a leaf subclass.
    
    A class's subclasses require final instantiation if it has at least one subclass,
    and all its subclasses have no further subclasses (i.e., they are leaf nodes).

    :param cls: The ontology class to check.
    :return: True if the class is a direct parent of a leaf class, else False.
    """
    subclasses = list(cls.subclasses())  # Get direct subclasses

    if not subclasses:  
        return False  # If the class itself is a leaf

    # Check if any direct subclasses are leaf classes

    return any(not list(sub.subclasses()) for sub in subclasses)
