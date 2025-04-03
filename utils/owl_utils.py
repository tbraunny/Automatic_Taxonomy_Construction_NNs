from owlready2 import *
from typing import List, Union, Optional, Any
import warnings
from builtins import TypeError
import traceback

from utils.annetto_utils import load_annetto_ontology

""" 1) Class Functions """


def get_highest_subclass_ancestor(cls: ThingClass) -> ThingClass:
    """
    Finds the highest (most general) superclass for a given class,
    Skips any parent that is not of type ThingClass.

    :param: cls: A ThingClass instance from which to find the highest ancestor.
    :return: The ThingClass of the highest ancestor of cls, or cls itself if no higher ancestor is found.
    """
    if not isinstance(cls, ThingClass):
        raise TypeError(f"The provided class '{cls}' is not a ThingClass.")
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


def get_class_parents(cls: ThingClass) -> List[ThingClass]:
    """
    Retrieves the direct 'is a' parent classes of a given ontology class.

    Note: No class in the ontology has more than one parent as of Feb 1 2025

    :param cls: The class for which to find direct parents.
    :return: A list of direct parent classes, excluding any restrictions.
    """
    if not isinstance(cls, ThingClass):
        raise TypeError(f"The provided class '{cls}' is not a ThingClass.")
    return [parent for parent in cls.is_a if isinstance(parent, ThingClass)]


def get_domain_class(
    ontology: Ontology, object_property: ObjectPropertyClass
) -> ThingClass:
    """
    Retrieves the domain class of a specified object property in an ontology.

    :param: ontology : The ontology object containing the property.
    :param: object_property : The name of the property whose domain class is to be retrieved.
    :returns: ThingClass: The domain class of the specified property if it exists, otherwise `None`.
    """
    if not isinstance(object_property, ObjectPropertyClass):
        raise TypeError(
            f"The provided property '{object_property}' is not an ObjectPropertyClass."
        )
    if not isinstance(ontology, Ontology):
        raise TypeError(f"The provided ontology '{ontology}' is not an Ontology.")
    # Get the domain class of the property
    domain_cls = object_property.domain
    if not domain_cls:
        print(f"Property '{object_property}' has no domain class.")
        return None
    return domain_cls


def get_onto_object_from_str(ontology: Ontology, class_name: str) -> ThingClass:
    """
    Retrieves a object from the ontology by its name.
    Can be used to retrieve classes, properties, etc.

    :param ontology: The ontology object.
    :param class_name: The name of the object to retrieve.
    :return: The object if found, otherwise None.
    """
    if not isinstance(class_name, str):
        raise TypeError(f"Class name '{class_name}' must be a string.")
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Ontology '{ontology}' must be an Owlready2 Ontology.")

    cls = getattr(ontology, class_name, None)
    if cls:
        return cls
    else:
        print(f"Class '{class_name}' not found in the ontology.")
        return None


def is_subclass_of_class(cls: ThingClass, parent_cls: ThingClass) -> bool:
    """
    Determines whether a given class is a subclass of another class.

    :param: cls: The class to check.
    :param: parent_cls: The class to check against.
    :return: True if cls is a subclass of parent_cls, False otherwise.
    """
    if not isinstance(cls, ThingClass):
        raise TypeError(f"The provided class '{cls}' is not a ThingClass.")
    if not isinstance(parent_cls, ThingClass):
        raise TypeError(
            f"The provided parent class '{parent_cls}' is not a ThingClass."
        )
    return issubclass(cls, parent_cls)


def is_subclass_of_any(cls: ThingClass) -> bool:
    """
    Determines whether a given class is a subclass of any class in the ontology.

    :param: cls: The class to check.
    :returns: True if cls is a subclass of any other class, False otherwise.
    """
    if cls.is_a:
        return True
    return False


def get_object_properties_with_domain_and_range(
    ontology: Ontology,
    domain_class: Union[ThingClass, Thing],
    range_class: Union[ThingClass, Thing],
) -> Optional[ObjectPropertyClass]:
    """
    Returns the object properties in the ontology that have the specified domain and range classes.

    :param ontology: The ontology.
    :param domain_class: The domain class to match.
    :param range_class: The range class to match.
    :return: The object property if found, otherwise None.
    """
    if not isinstance(domain_class, ThingClass) and not isinstance(domain_class, Thing):
        raise TypeError(f"Invalid domain class type '{domain_class}'.")
    if not isinstance(range_class, ThingClass) and not isinstance(range_class, Thing):
        raise TypeError(f"Invalid range class type '{range_class}'.")
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Invalid ontology '{ontology}'.")
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
    cls: Union[ThingClass, Thing], ontology: Ontology
) -> Optional[List[ThingClass]]:
    """
    Retrieves classes connected to the given class via object properties.

    :param: cls: The class for which to find connected classes.
    :param: ontology: The ontology containing the classes.
    :return: A list of connected classes, or None if no connections are found.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class type '{cls}'.")
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Invalid ontology type '{ontology}'.")
    connected_classes = set()
    object_properties = [
        prop for prop in ontology.object_properties() if cls in prop.domain
    ]

    for prop in object_properties:
        for range_cls in prop.range:
            # Skip if the range is the same as cls
            if range_cls == cls:
                continue
            if isinstance(range_cls, ThingClass):
                connected_classes.add(range_cls)
    connected_classes = list(connected_classes)

    return connected_classes if connected_classes != [] else None


def get_immediate_subclasses(
    cls: Union[ThingClass, Thing],
) -> Optional[List[ThingClass]]:
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
    ontology: Ontology, cls: Union[ThingClass, Thing]
) -> Optional[List[DataPropertyClass]]:
    """
    Retrieves all data properties in the given ontology that have the specified class as their domain.

    :param: ontology: The ontology containing the properties.
    :param: cls: The class for which to find properties with this domain.
    :return: A list of data properties that have the specified class as their domain if any, else return None.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class type '{cls}'.")
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Invalid ontology type '{ontology}'.")
    return [prop if cls in prop.domain else None for prop in ontology.data_properties()]
    # return [prop for prop in ontology.data_properties() if cls in prop.domain]


def get_class_object_properties(
    ontology: Ontology, cls: Union[ThingClass, Thing]
) -> Optional[List[ObjectPropertyClass]]:
    """
    Retrieves all object properties in the given ontology that have the specified class as their domain.

    :param: ontology: The ontology containing the properties.
    :param: cls: The class for which to find properties with this domain.
    :return: A list of object properties that have the specified class as their domain if any, else None.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class type '{cls}'.")
    if not isinstance(ontology, Ontology):
        raise TypeError(f"Invalid ontology type '{ontology}'.")
    return [
        prop if cls in prop.domain else None for prop in ontology.object_properties()
    ]
    # return [prop for prop in ontology.object_properties() if cls in prop.domain]


def get_class_restrictions(
    cls: Union[ThingClass, Thing],
) -> Optional[List[Restriction]]:
    """
    Retrieves all restrictions (including cardinality restrictions) applied to a specified class.

    :param: cls: The class for which to retrieve restrictions.
    :return: A list of restrictions applied to the class.
    """
    if not isinstance(cls, (Thing, ThingClass)):
        raise TypeError(f"Invalid class type '{cls}'.")
    return [
        restriction if isinstance(restriction, Restriction) else None
        for restriction in cls.is_a
    ]
    # return [restriction for restriction in cls.is_a if isinstance(restriction, Restriction)]


def create_class(
    ontology: Ontology, class_name: str, base_class: ThingClass
) -> ThingClass:
    """
    Dynamically creates a class in the ontology while ensuring name uniqueness.

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
    ontology: Ontology, subclass_name: str, parent_class: ThingClass
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


def get_class_instances(cls: ThingClass) -> Optional[List[Thing]]:
    """
    Retrieves all instances of the provided class.

    :param: cls: The class for which to retrieve its instances.
    :return: A list of instances of the specified class if any, else None.
    """
    if not isinstance(cls, ThingClass):
        raise TypeError(
            f"The provided class '{cls}' is not a ThingClass.", exec_info=True
        )
    return cls.instances() if cls.instances() else None


def get_instantiated_properties(instance: Thing) -> List[Property]:
    """
    Retrieves all properties that have been set (instantiated) on the specified instance.

    :param: instance: The instance for which to retrieve instantiated properties.
    :return: A list of properties that have been set on the instance.
    """
    if not isinstance(instance, Thing):
        raise TypeError(
            f"The provided instance '{instance}' is not a Thing.", exec_info=True
        )
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

    :param: instance: The instance for which to retrieve property values.
    :return: A dictionary where keys are property names and values are the values set for each property.
    """
    if not isinstance(instance, Thing):
        raise TypeError(
            f"The provided instance '{instance}' is not a Thing.", exec_info=True
        )
    property_values = {}
    for prop in get_instantiated_properties(instance):
        property_values[prop.name] = instance.__getattr__(prop.name)
    return property_values


def create_cls_instance(
    ontology: Ontology, cls: ThingClass, instance_name: str, **properties
) -> Thing:
    """
    Creates an instance of a given class.

    :param cls: The class to instantiate.
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
    instance = cls(instance_name, namespace=ontology)

    # Assert that the instance is of the correct class
    if not isinstance(instance, cls):
        raise TypeError(
            f"Failed to create instance '{instance_name}' of class '{cls}'."
        )
    if not isinstance(instance, Thing):
        raise TypeError(f"Failed to create instance '{instance_name}' of type Thing.")

    # Assign additional properties if provided
    # TODO: Not sure if this is the correct way to set properties
    if properties:
        try:
            for key, value in properties.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
                else:
                    print(f"Warning: {cls.name} has no property '{key}'")

        except Exception as e:
            print(f"Error setting properties for {cls.name}: {e}")

    return instance


def get_instance_class(instance: Thing) -> ThingClass:
    """
    Retrieves the ThingClass of a given instance.

    :param instance: The instance for which to retrieve the class.
    :return: The class of the instance.
    """
    if not isinstance(instance, Thing):
        raise TypeError(
            f"The provided instance '{instance}' is not a valid Thing.", exec_info=True
        )
    return type(instance)


def assign_object_property_relationship(
    domain: Union[ThingClass, Thing],
    range: Union[ThingClass, Thing],
    object_property: ObjectPropertyClass,
):
    """
    Connect two classes via a specified ObjectPropertyClass.

    :param domain: The Thing instance representing the domain.
    :param range: The Thing instance representing the range.
    :param object_property: The ObjectPropertyClass to connect the instances.
    """
    # Check if domain and ranges are instances of Thing
    if not isinstance(domain, (Thing, ThingClass)):
        raise TypeError(
            f"The 'domain' argument '{domain}' '{type(domain)} must be an instance of Thing or ThingClass."
        )
    if not isinstance(range, (Thing, ThingClass)):
        raise TypeError(
            f"The 'range' argument '{domain}' must be an instance of Thing or ThingClass."
        )
    if not isinstance(object_property, ObjectPropertyClass):
        raise TypeError(
            f"The 'object_property' argument '{object_property}' '{type(object_property)}' must of type ObjectPropertyClass."
        )
    # Connect the two Thing instances
    object_property[domain].append(range)

    # Check if the connection was successful
    if not range in object_property[domain]:
        raise ValueError(
            f"Failed to connect {domain} to {range} via {object_property}",
            exec_info=True,
        )


def link_data_property_to_instance(
    instance: Thing, data_property: DataPropertyClass, value: Any
):
    """
    Links a data property to a Thing instance with a specified value.

    :param instance: The Thing instance to link the data property to.
    :param data_property: The DataPropertyClass to link to the instance.
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
    # Set the data property value for the instance
    instance.data_property = [value]

    # Check if the value was set successfully
    if not instance.data_property == value:
        raise ValueError(
            f"Failed to set data property '{data_property}' to '{value}' for instance '{instance}'."
        )


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

    with ontology:
        p3 = create_class(ontology, "Person", base_class=ontology.Network)
        p2 = create_class(ontology, "Person2", base_class=ontology.TaskCharacterization)
        p4 = create_subclass(ontology, "Person", p2)
        print(p3, p2, p4)

    # instance1 = create_cls_instance(ontology.Network, "Conv Network")
    # instance2 = create_cls_instance(ontology.CostFunction, "Class1")

    # classes = ontology.CostFunction.is_a
    # print(classes)

    # classes = assign_object_property_relationship(ontology, instance1, instance2)
