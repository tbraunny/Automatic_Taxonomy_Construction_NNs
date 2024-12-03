from owlready2 import *

def requires_final_instantiation(cls, ontology):
    """
    Determines if a class requires final instantiation as a standalone object.

    A class requires final instantiation if it:
    - Has no data properties.
    - Has no object properties.

    Args:
        cls (ThingClass): the class to check.
        ontology (Ontology): the ontology to which the class belongs.

    Returns:
        bool: True if the class must be instantiated as a final object, otherwise False.
    """
    has_data_properties = any(cls in prop.domain for prop in ontology.data_properties())
    has_object_properties = any(cls in prop.domain for prop in ontology.object_properties())
    return not has_data_properties and not has_object_properties

def get_connected_classes(cls, ontology):
    """
    Retrieves classes connected to the given class via object properties.

    Args:
        cls (ThingClass): the class to explore.
        ontology (Ontology): the ontology to which the class belongs.

    Returns:
        set: A set of connected ThingClasses.
    """
    connected_classes = set()
    obj_props = [prop for prop in ontology.object_properties() if cls in prop.domain]
    for prop in obj_props:
        for range_cls in prop.range:
            if isinstance(range_cls, ThingClass):
                connected_classes.add(range_cls)
    return connected_classes


def process_class(cls, ontology, visited_classes):
    """
    Processes a class, retrieves its properties, and recursively processes related classes.
    Yields classes that require instantiation.

    Args:
        cls (ThingClass): the class to process.
        ontology (Ontology): the ontology to which the class belongs.
        visited_classes (set): set of already visited classes to avoid duplication.

    Yields:
        ThingClass: classes that require instantiation.
    """
    if cls in visited_classes:
        return

    visited_classes.add(cls)

    # Determine if the class requires final instantiation
    if requires_final_instantiation(cls, ontology):
        yield cls
    else:
        # Process data properties (if needed)
        data_props = [prop for prop in ontology.data_properties() if cls in prop.domain]

        # Process object properties and their range classes
        connected_classes = get_connected_classes(cls, ontology)

        for connected_cls in connected_classes:
            if connected_cls not in visited_classes:
                yield from process_class(connected_cls, ontology, visited_classes)

def process_subclasses(cls, ontology, visited_classes):
    """
    Recursively processes subclasses of a class and yields classes that require instantiation.

    Args:
        cls (ThingClass): the class whose subclasses to process.
        ontology (Ontology): the ontology to which the class belongs.
        visited_classes (set): set of already visited classes to avoid duplication.

    Yields:
        ThingClass: subclasses that require instantiation.
    """
    if cls in visited_classes:
        return

    visited_classes.add(cls)

    # Retrieve all direct subclasses of the current class
    subclasses = cls.subclasses()

    for subclass in subclasses:
        if subclass in visited_classes:
            continue  # Skip already visited subclasses

        # Add the subclass to the visited set
        visited_classes.add(subclass)

        # Determine if the subclass requires final instantiation
        if requires_final_instantiation(subclass, ontology):
            yield subclass
        else:
            # Recursively process subclasses of this subclass
            yield from process_subclasses(subclass, ontology, visited_classes)



def traverse_ontology(ontology, start_class_name):
    """
    Traverses the ontology starting from the given class name.

    Args:
        ontology (Ontology): the ontology to traverse.
        start_class_name (str): the name of the starting class.

    Yields:
        ThingClass: classes that require instantiation.
    """
    start_class = ontology[start_class_name]
    if start_class is None:
        raise ValueError(f"Class '{start_class_name}' does not exist in the ontology.")

    visited_classes = set()
    yield from process_class(start_class, ontology, visited_classes)
