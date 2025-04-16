from owlready2 import *
from typing import List, Union, Optional, Any, Type
import warnings
from builtins import TypeError

from utils.annetto_utils import load_annetto_ontology

""" 1) Class Functions """


def get_highest_subclass_ancestor(cls: ThingClass) -> ThingClass:
    """
    Finds the highest (most general) superclass for a given class,
    Skips any parent that is not of type ThingClass.

    Args:
        cls: A ThingClass instance from which to find the highest ancestor.

    Returns:
        The ThingClass of the highest ancestor of cls, or cls itself
        if no higher ancestor is found.
    """
    current = cls
    while True:
        # Get immediate superclasses excluding owl.Thing
        superclasses = []
        for parent in current.is_a:
            if not isinstance(parent, ThingClass):
                continue  # Skip if not a ThingClass

            if parent == owl.Thing:
                continue  # Skip owl.Thing

            superclasses.append(parent)

        if not superclasses:
            # No more valid parents to traverse
            return current

        # Move to the first valid superclass
        current = superclasses[0]


def get_class_parents(cls: ThingClass) -> list:
    """
    Retrieves the direct 'is a' parent classes of a given ontology class.

    Note: No class in the ontology has more than one parent as of Feb 1 2025

    :param cls: The ontology class (ThingClass) for which to find direct parents.
    :return: A list of direct parent classes, excluding any restrictions.
    """
    return [parent for parent in cls.is_a if isinstance(parent, ThingClass)]


def get_domain_class(ontology: Ontology, property_name: str) -> ThingClass:
    """
    Retrieves a class's object property domain class for a given class in an ontology.
    """

    # Find the property in the ontology
    prop = getattr(ontology, property_name, None)
    if not prop:
        print(f"Property '{property_name}' not found in the ontology.")
        return None

    # Get the domain class of the property
    domain_cls = prop.domain
    if not domain_cls:
        print(f"Property '{property_name}' has no domain class.")
        return None

    return domain_cls


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


def is_subclass_of_class(cls: Union[ThingClass,Thing], parent_cls: ThingClass) -> bool:
    """
    Determines whether a given class is a subclass of another class.

    Args:
        cls: The class to check.
        parent_cls: The class to check against.

    Returns:
        bool: True if cls is a subclass of parent_cls, False otherwise.
    """
    if not isinstance(cls, (ThingClass, Thing)):
        raise TypeError(f"The provided class '{cls}' is not a ThingClass or Thing.")
    if not isinstance(parent_cls, ThingClass):
        raise TypeError(
            f"The provided parent class '{parent_cls}' is not a ThingClass."
        )
    return issubclass(cls, parent_cls)


def is_subclass_of_any(ontology, cls):
    """
    Determines whether a given class is a subclass of any class in the ontology.

    Args:
        ontology: The ontology containing the classes.
        cls: The class to check.

    Returns:
        bool: True if cls is a subclass of any other class, False otherwise.
    """

    if cls.is_a:
        return True
    return False


def get_base_class(onto: Ontology):
    return onto.ANNConfiguration


def get_object_properties_with_domain_and_range(ontology, domain_class, range_class):
    """
    Returns a list of object properties in the ontology that have the specified domain and range classes.

    :param ontology: The loaded Owlready2 ontology.
    :param domain_class: The domain class to match.
    :param range_class: The range class to match.
    :return: A list of matching object properties.
    """
    matching_property = None

    counter = 0

    # Iterate over all object properties in the ontology
    for obj_property in ontology.object_properties():
        # Get domain and range for the property
        domains = list(obj_property.domain)
        ranges = list(obj_property.range)

        # Check if the domain and range match the given classes
        if domain_class in domains and range_class in ranges:
            matching_property = obj_property
            counter += 1

    if counter > 1:
        raise ValueError(
            f"More than one object property found with domain ({domain_class}) and range classes ({range_class})."
        )

    return matching_property


def get_connected_classes(
    cls: ThingClass, ontology, return_object_properties: bool = False
):
    """
    Retrieves classes connected to the given class or instance via object properties.

    :param: cls: The class or instance for which to find connected classes.
    :param: ontology: The ontology containing the classes.
    :return: A list of connected classes, or None if no connections are found.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class or instance type for '{cls}'.")
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Invalid ontology type for '{cls}'.")

    connected_classes = set()

    if isinstance(cls, ThingClass):
        # For ThingClass, use the domain of object properties
        object_properties = [
            prop for prop in ontology.object_properties() if cls in prop.domain
        ]
        for prop in object_properties:
            for range_cls in prop.range:
                if range_cls != cls and isinstance(range_cls, ThingClass):
                    connected_classes.add(range_cls)
        return list(connected_classes) if connected_classes else None

    elif isinstance(cls, Thing):
        for prop in cls.get_properties():
            if isinstance(prop, ObjectPropertyClass):  # Ensure it's an object property
                values = cls.__getattr__(prop.name)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, Thing):
                            print(value)
                            connected_classes.add(value)
                elif isinstance(values, Thing):
                    print(values)
                    connected_classes.add(values)

        return list(connected_classes) if connected_classes else None


def get_immediate_subclasses(cls: ThingClass) -> List[ThingClass]:
    """
    Retrieves all direct subclasses of a given class.

    :param: cls: The class for which to find it's subclasses.
    :return: A list of subclasses to the given class if any, else return None.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class type '{cls}'.")

    return list(cls.subclasses()) if cls.subclasses() else None


def get_all_subclasses(
    cls: Union[ThingClass, Thing], visited=None
) -> Optional[List[ThingClass]]:
    """
    Recursively retrieves all unique subclasses of a given class.

    :param: cls: The class for which to find its subclasses.
    :return: A list of all subclasses of the given class, including nested ones, if any, else, return None.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class type '{cls}'.")

    if visited is None:  # initial
        visited = set()
    if cls in visited:  # if node has been visited, skip
        return []
    visited.add(cls)  # mark as visited

    subclasses = set(get_immediate_subclasses(cls))

    if not subclasses:
        return []

    for subclass in list(subclasses):
        subclasses.update(
            get_all_subclasses(subclass, visited)
        )  # find all subclasses thru RECURSION

    else:
        return list(subclasses)


def get_class_data_properties(
    ontology: Ontology, onto_class: ThingClass
) -> List[Property]:
    """
    Retrieves all data properties in the given ontology that have the specified class as their domain.

    Args:
        ontology (Ontology): The ontology containing the properties.
        onto_class (ThingClass): The class for which to find properties with this domain.

    Returns:
        List[Property]: A list of data properties that have the specified class as their domain.
    """
    return [prop for prop in ontology.data_properties() if onto_class in prop.domain]


def get_class_object_properties(
    ontology: Ontology, onto_class: ThingClass
) -> List[Property]:
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
    return [
        restriction
        for restriction in onto_class.is_a
        if isinstance(restriction, Restriction)
    ]


def create_class(
    ontology: Ontology, class_name: str, base_class: ThingClass = None
) -> ThingClass:
    """
    Dynamically creates a class in the ontology if it does not already exist.

    :param ontology: The ontology in which to create the class.
    :param class_name: The desired name of the class.
    :param base_class: The base class for the new class.
    :return: The newly created class or the existing class if it already exists.
    """
    if not isinstance(class_name, str):
        raise TypeError(f"Class name '{class_name}' must be a string.", exec_info=True)
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Ontology '{ontology}' must be an Ontology.", exec_info=True)
    if not isinstance(base_class, ThingClass):
        raise TypeError(
            f"Base class '{base_class}' must be a ThingClass.", exec_info=True
        )

    # Check if a class or property with the same name already exists
    existing_entity = getattr(ontology, class_name, None)

    if existing_entity:
        if isinstance(existing_entity, ThingClass):
            warnings.warn(f"Class '{class_name}' already exists, reusing it.")
            return existing_entity
        else:
            # If an entity of a different type exists, rename the new class
            new_class_name = f"{class_name}_c"
            warnings.warn(
                f"Name conflict: '{class_name}' exists as a different type. Renaming to '{new_class_name}'."
            )
            class_name = new_class_name

    # Dynamically create the new class
    new_class = type(class_name, (base_class,), {"namespace": ontology})

    return new_class


def create_subclass(
    ontology: Ontology, class_name: str, base_class: ThingClass
) -> ThingClass:
    """
    Dynamically creates a subclass in the ontology.

    :param ontology: The ontology in which to create the subclass.
    :param subclass_name: The name of the subclass.
    :param parent_class: The parent class for the subclass.
    :return: The newly created subclass or the existing one if it already exists.
    """
    if not isinstance(subclass_name, str):
        raise TypeError(
            f"Subclass name '{subclass_name}' must be a string.", exec_info=True
        )
    if not isinstance(parent_class, ThingClass):
        raise TypeError(
            f"Parent class '{parent_class}' must be ThingClass.",
            exec_info=True,
        )
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Ontology '{ontology}' must be an Ontology.", exec_info=True)

    # Check if the class already exists
    existing_class = getattr(ontology, subclass_name, None)

    if existing_class:
        if issubclass(existing_class, parent_class):
            warnings.warn(f"Subclass '{subclass_name}' already exists, reusing it.")
            return existing_class
        else:
            new_subclass_name = f"{subclass_name}_sub"
            warnings.warn(
                f"Name conflict: '{subclass_name}' exists but is not a subclass. Renaming to '{new_subclass_name}'."
            )
            subclass_name = new_subclass_name

    # Create the new subclass
    new_subclass = type(subclass_name, (parent_class,), {"namespace": ontology})

    return new_subclass


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


def get_instance_class_properties(
    ontology: Ontology, instance: Thing
) -> List[Property]:
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
        if instance.__getattr__(
            prop.name
        ):  # Check if the property has a non-empty value
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


def get_object_properties_for_class(ontology: Ontology, cls: ThingClass):
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


def create_cls_instance(
    onto_class: ThingClass, instance_name: str, **properties
) -> Thing:
    """
    Creates an instance of a given class.

    :param onto_class: The class to instantiate.
    :param instance_name: Name of the new instance.
    :param properties: (Optional) Additional properties to set (as keyword arguments).
    :return: The created instance.
    """
    if not isinstance(instance_name, str):
        raise TypeError("The instance name must be a string.")
    if not isinstance(cls, ThingClass):
        raise TypeError(f"The provided class {cls} is not a valid ThingClass.")

    # Check if the class already exists
    existing_class = getattr(ontology, instance_name, None)

    if existing_class:
        if existing_class in cls.instances():
            warnings.warn(f"Instance '{instance_name}' already exists, reusing it.")
            return existing_class
        else:
            new_instance_name = f"{instance_name}_inst"
            warnings.warn(
                f"Name conflict: '{instance_name}' exists but is not an instance of {cls}. Renaming to '{new_instance_name}'."
            )
            instance_name = new_instance_name

    # Create instance with name
    instance = onto_class(instance_name)

    # Assign additional properties if provided
    for key, value in properties.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            print(f"Warning: {onto_class.name} has no property '{key}'")

    return instance


def get_instance_class(instance: Thing) -> ThingClass:
    return type(instance)


def assign_object_property_relationship(
    domain: Thing, ranges: Thing, object_property: ObjectPropertyClass
):
    """
    Connect two Thing instances via a specified ObjectProperty in Owlready2.

    :param domain: The Thing instance representing the domain.
    :param ranges: The Thing instance representing the range.
    :param object_property: The ObjectProperty to connect the instances.
    """

    # Check if domain and ranges are instances of Thing
    if not isinstance(domain, Thing):
        raise TypeError("The 'domain' argument must be an instance of Thing.")
    if not isinstance(ranges, Thing):
        raise TypeError("The 'ranges' argument must be an instance of Thing.")

    # Check if object_property is a valid ObjectProperty
    if not isinstance(object_property, ObjectPropertyClass):
        raise TypeError(
            "The 'object_property' argument must be an instance of ObjectProperty."
        )

    # Connect the two Thing instances
    try:
        object_property[domain].append(range)
    except Exception as e:
        raise ValueError(
            f"Failed to assign object property {object_property} from {domain} to {range} (perhaps because it is functional?): {e}"
        )

    # Check if the connection was successful
    related_objects = list(object_property[domain])
    if range not in related_objects:

        raise ValueError(
            f"Verification failed: {range} not found in {object_property}[{domain}]. Current values: {related_objects}"
        )
    
def is_functional_property(property: DataPropertyClass) -> bool:
    """
    Determines if a given property is functional.
    :param property: The DataPropertyClass to check.
    :return: True if the property is functional, False otherwise.
    """
    return isinstance(property, FunctionalProperty)

def link_data_property_to_instance(
    instance: Thing, data_property: DataPropertyClass, value: Any
):
    """
    Links a data property to a Thing instance with a specified value.

    :param instance: The Thing instance to link the data property to.
    :param data_property: The DataProperty to link to the instance.
    :param value: The value to set for the data property.
    """
    # Check if instance is an instance of Thing
    if not isinstance(instance, Thing):
        raise TypeError("The 'instance' argument must be an instance of Thing.")
    # Check if data_property is a valid DataPropertyClass
    if not isinstance(data_property, DataPropertyClass):
        raise TypeError(
            "The 'data_property' argument must be an instance of DataPropertyClass."
        )
    # setattr(instance, data_property.name, value)
    setattr(instance, data_property.name, [value] if not isinstance(value, list) else value)

    # # Verify the value was set correctly
    # if value not in getattr(instance, data_property.name, []):
    #     raise ValueError(
    #         f"Failed to set data property '{data_property.name}' to '{value}' for instance '{instance}'."
    #     )
    
    if is_functional_property(data_property):
        # Assign the value directly for functional properties
        setattr(instance, data_property.name, value)
    else:
        # Assign the value as a list for non-functional properties
        setattr(instance, data_property.name, [value] if not isinstance(value, list) else value)
    # # Set the data property value for the instance
    # instance.data_property = [value]

    # # Check if the value was set successfully
    # if not instance.data_property == value:
    #     raise ValueError(
    #         f"Failed to set data property '{data_property}' to '{value}' for instance '{instance}'."
    #     )


def create_class_data_property(
    ontology: Ontology,
    property_name: str,
    domain_class: Union[ThingClass, Thing],
    range_type,
    functional=False,
) -> DataPropertyClass:
    """
    Dynamically creates a DataProperty for a class.
    Must still assign the value of the property to an instance of the domain class.
    A functional property is one where each individual has at most one value for the property.

    :param ontology: The ontology
    :param property_name: Name of the DataProperty to be created
    :param domain_class: Class that serves as the domain
    :param range_type: Data type (int, float, str, bool) of the property
    :param functional: Boolean flag to set the property as functional
    :return: Created DataProperty class
    """
    if hasattr(ontology, property_name): # need new logic to check if property exits for the given domain class
        print(f"Property '{property_name}' already exists in the ontology. Error or unexepcted behavior may occur.")
    valid_range_types = {int, float, str, bool}
    if range_type not in valid_range_types:
        raise ValueError(
            f"Invalid range_type: {range_type}. Must be one of {valid_range_types}"
        )

    if functional:
        NewDataProperty = type(
            property_name,
            (DataProperty, FunctionalProperty),
            {"namespace": ontology, "domain": [domain_class], "range": [range_type]},
        )
    else:
        NewDataProperty = type(
            property_name,
            (DataProperty,),
            {"namespace": ontology, "domain": [domain_class], "range": [range_type]},
        )

    return NewDataProperty

def create_class_object_property(
    ontology: Ontology,
    property_name: str,
    domain_class: Union[ThingClass, Thing],
    range_class: Union[ThingClass, Thing],
) -> Type[ObjectProperty]:
    """
    Dynamically creates an ObjectProperty for a class.
    Must still assign the value of the property to an instance of the domain class.

    A functional ObjectProperty means each subject can be linked to at most one object via the property.

    :param ontology: The ontology
    :param property_name: Name of the ObjectProperty to be created
    :param domain_class: Class that serves as the domain
    :param range_class: Class that serves as the range
    :return: Created ObjectProperty class
    """

    # if functional:
    NewObjectProperty = type(
        property_name,
        (ObjectProperty, FunctionalProperty),
        {
            "namespace": ontology,
            "domain": [domain_class],
            "range": [range_class],
        },
    )
    # else:
    #     NewObjectProperty = type(
    #         property_name,
    #         (ObjectProperty,),
    #         {
    #             "namespace": ontology,
    #             "domain": [domain_class],
    #             "range": [range_class],
    #         },
    #     )

    return NewObjectProperty

def is_functional_property_for(domain: Union[ThingClass, Thing], property_: Union[ObjectPropertyClass, DataPropertyClass]) -> bool:
    """
    Check if an ObjectProperty or DataProperty is functional for a given domain instance or class.

    :param domain: The domain (Thing or ThingClass) to check against.
    :param property_: The property (ObjectProperty or DataProperty) to inspect.
    :return: True if functional, False otherwise.
    """
    if not isinstance(domain, (Thing, ThingClass)):
        raise TypeError(f"'domain' must be a Thing or ThingClass, got {type(domain)}")

    if not isinstance(property_, (ObjectPropertyClass, DataPropertyClass)):
        raise TypeError(f"'property_' must be an ObjectPropertyClass or DataPropertyClass, got {type(property_)}")

    # Check whether the property is functional at the ontology level
    if issubclass(property_, FunctionalProperty):
        return True

    # In Owlready2, a property can also be declared functional per domain-class via .is_functional_for
    try:
        return property_.is_functional_for(domain)
    except AttributeError:
        return False

def entitiy_exists(ontology: Ontology, name: str) -> bool:
    """
    Check if an entity (class, property, or instance) exists in the ontology.
    :param ontology: The ontology to check.
    :param name: The name of the entity to check.
    :return: True if the entity exists, False otherwise.
    """
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Ontology '{ontology}' must be an Ontology.")
    return hasattr(ontology, name)

def list_owl_classes(onto: Ontology) -> List[ThingClass]:
    # Return list of all classes in ontology
    return [cls for cls in onto.classes()]


def list_owl_object_properties(onto: Ontology) -> List[ObjectPropertyClass]:
    # return all object properties of the ontology
    return [prop for prop in onto.object_properties()]


def list_owl_data_properties(onto: Ontology) -> List[DataPropertyClass]:
    # return all data properties of the ontology
    return [prop for prop in onto.data_properties()]


if __name__ == "__main__":

    ontology = load_annetto_ontology("test")

    list_owl_props = list_owl_data_properties(ontology)
    print()


    # with ontology:
    #     p3 = create_class(ontology, "Person", base_class=ontology.Network)
    #     p2 = create_class(ontology, "Person2", base_class=ontology.TaskCharacterization)
    #     p4 = create_subclass(ontology, "Person", p2)
    #     print(p3, p2, p4)

    # instance1 = create_cls_instance(ontology.Network, "Conv Network")
    # instance2 = create_cls_instance(ontology.CostFunction, "Class1")

    # classes = ontology.CostFunction.is_a
    # print(classes)

    # classes = assign_object_property_relationship(ontology, instance1, instance2)
