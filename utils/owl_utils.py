from owlready2 import *
from typing import List
import warnings

def load_ontology(ontology_path):
    return get_ontology(ontology_path).load()

""" 1) Class Functions """

def get_class_parents(cls: ThingClass) -> list:
    """
    Retrieves the direct 'is a' parent classes of a given ontology class.

    Note: No class in the ontology has more than one parent as of Feb 1 2025
    
    :param cls: The ontology class (ThingClass) for which to find direct parents.
    :return: A list of direct parent classes, excluding any restrictions.
    """
    return [parent for parent in cls.is_a if isinstance(parent, ThingClass)]

def get_class_by_name(onto, class_name):
    """
    Retrieves a class object from an ontology by its name.

    Args:
        onto: The ontology object.
        class_name (str): The name of the class to retrieve.

    Returns:
        The class object if found, or None if not found.
    """
    cls = getattr(onto, class_name, None)
    if cls:
        return cls
    else:
        print(f"Class '{class_name}' not found in the ontology.")
        return None


def is_subclass(cls, parent_cls):
    """
    Determines whether a given class is a subclass of another class.

    !!! Not sure if this is the correct logic !!!
    Args:
        cls: The class to check.
        parent_cls: The parent class to compare against.

    Returns:
        bool: True if cls is a subclass of parent_cls, False otherwise.
    """
    return parent_cls in cls.ancestors() and cls is not parent_cls


def get_base_class(onto:Ontology):
    return onto.ANNConfiguration


def get_connected_classes(cls, ontology):
    """
    Retrieves classes connected to the given class via object properties.
    """
    connected_classes = set()
    object_properties = [prop for prop in ontology.object_properties() if cls in prop.domain]
    for prop in object_properties:
        for range_cls in prop.range:
            # Skip if the range is the same as cls
            if range_cls == cls:
                continue
            if isinstance(range_cls, ThingClass):
                connected_classes.add(range_cls)
    connected_classes = list(connected_classes)
    return connected_classes if connected_classes != [] else None



def get_subclasses(cls):
    """
    Retrieves all subclasses of a given class.

    Args:
        cls (ThingClass): The class for which to find it's subclasses.

    Returns:
        List[ThingClass]: A list of subclasses to the given class.
    """
    return list(cls.subclasses())



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



def get_class_data_properties(ontology: Ontology, onto_class: ThingClass) -> List[Property]:
    """
    Retrieves all data properties in the given ontology that have the specified class as their domain.

    Args:
        ontology (Ontology): The ontology containing the properties.
        onto_class (ThingClass): The class for which to find properties with this domain.

    Returns:
        List[Property]: A list of data properties that have the specified class as their domain.
    """
    return [prop for prop in ontology.data_properties() if onto_class in prop.domain]



def get_class_object_properties(ontology: Ontology, onto_class: ThingClass) -> List[Property]:
    """
    Retrieves all object properties in the given ontology that have the specified class as their domain.

    Args:
        ontology (Ontology): The ontology containing the properties.
        onto_class (ThingClass): The class for which to find properties with this domain.

    Returns:
        List[Property]: A list of object properties that have the specified class as their domain.
    """
    return [prop for prop in ontology.object_properties() if onto_class in prop.domain]



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

def get_class_instances(cls: ThingClass) -> list:
    """
    Retrieves all instances of the provided class.

    Args:
        cls (ThingClass): The class for which to retrieve its instances.

    Returns:
        List[Thing]: A list of instances (instances) of the specified class.
    """
    return cls.instances()

def explore_instance(instance, depth=0, visited=None):
    """
    Recursively explores an OWL instance and its relationships, printing its properties and values.

    Args:
        instance (Thing): The starting instance to explore. Defaults to `onto.GAN`.
        depth (int): The current depth of recursion, used for indentation. Defaults to 0.
        visited (set): A set of visited instances to prevent infinite loops during recursion. Defaults to None.

    Returns:
        None: The function prints the exploration results and does not return any value.
    """
    
    # Initialize the visited set if this is the first call
    if visited is None:
        visited = set()
    
    # Prevent infinite loops by skipping already visited instances
    if instance in visited:
        return
    visited.add(instance)

    # Print the instance's name with indentation based on depth
    indent = "  " * depth
    print(f"{indent}instance: {instance.name}")

    # Loop through all properties of the instance
    for prop in instance.get_properties():
        print(f"{indent}  Property: {prop.name}")
        
        # Iterate through the values of the property
        for value in prop[instance]:
            try:
                # If the value is another instance, recursively explore it
                if isinstance(value, Thing):
                    explore_instance(value, depth + 1, visited)
                else:  # Otherwise, print the literal value
                    print(f"{indent}    Value: {value}")
            except Exception as e:
                # Handle errors gracefully and log them
                print(f"{indent}    Error reading value: {e}")


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
    for instance in ontology.instances():
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

# Create Instances

def create_cls_instance(onto_class: ThingClass, instance_name:str, **properties):
    """
    Creates an instance of a given class.

    :param onto_class: The class to instantiate.
    :param instance_name: Name of the new instance.
    :param properties: (Optional) Additional properties to set (as keyword arguments).
    :return: The created instance.
    """
    if not issubclass(onto_class, Thing):
        raise ValueError("The provided class must be a subclass of owlready2.Thing")
    
    if not instance_name:
        print("Warning: {onto_class} can't be instantiated without a name.")
        return

    # Create instance with name
    instance = onto_class(instance_name)

    # Assign additional properties if provided
    for key, value in properties.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            print(f"Warning: {onto_class.__name__} has no property '{key}'")

    return instance

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