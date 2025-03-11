from owlready2 import *
from typing import List
import warnings


def load_ontology(ontology_path):
    return get_ontology(ontology_path).load()


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


def is_subclass_of_class(cls, parent_cls):
    """
    Determines whether a given class is a subclass of another class.

    Args:
        cls: The class to check.
        parent_cls: The class to check against.

    Returns:
        bool: True if cls is a subclass of parent_cls, False otherwise.
    """
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
    Retrieves classes connected to the given class via object properties.
    """
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


def get_immediate_subclasses(cls: ThingClass) -> List[ThingClass]:
    """
    Retrieves all direct subclasses of a given class.

    Args:
        cls (ThingClass): The class for which to find it's subclasses.

    Returns:
        List[ThingClass]: A list of subclasses to the given class.
    """
    return list(cls.subclasses())


def get_all_subclasses(cls: ThingClass , visited=None) -> List[ThingClass]:
    """
    # Recursively retrieves all unique subclasses of a given class.

    # Args:
    #     cls (ThingClass): The class for which to find its subclasses.
    #     visited: Set of visited nodes

    # Returns:
    #     List[ThingClass]: A list of all subclasses of the given class, including nested ones.
    # """

    if visited is None: #initial
        visited = set()

    if cls in visited: # if node has been visited, skip
        return []

    visited.add(cls)  # mark as visited
    subclasses = set(get_immediate_subclasses(cls))

    for subclass in list(subclasses): 
        subclasses.update(get_all_subclasses(subclass, visited)) # find all subclasses thru RECURSION

    return list(subclasses)



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

    Args:
        ontology (Ontology): The ontology in which to create the class.
        class_name (str): The name of the class to create.
        base_class (ThingClass, optional): The base class for the new class. Defaults to None, which uses Thing.

    Returns:
        ThingClass: The newly created class or the existing class if it already exists.
    """
    # Check if the class already exists

    ontology_classes:List[ThingClass] = list_owl_classes(ontology)
    if class_name in ontology_classes:
        warnings.warn(f"Class '{class_name}' already exists, continuing.")
        return class_name

    # Set base class to Thing if no base_class is provided
    if base_class is None:
        base_class = ontology.Thing

    # Dynamically create the new class using `type()`
    new_class = type(class_name, (base_class,), {"namespace": ontology})
    setattr(
        ontology, class_name, new_class
    )  # Add the new class to the ontology's namespace
    return new_class


def create_subclass(
    ontology: Ontology, class_name: str, base_class: ThingClass
) -> ThingClass:
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

def find_instance_properties(instance, has_property=[], equals=[], found=[], visited=None):
    '''
    Finds all properties based on passed in has_properties
    
    Args:
        instance (ThingClass): the class for which is an instance
        has_property: the has properties we are looking for. This is typically edge properties
        equals: a set of dictionary that define some comparison operations
        found: a list of found elements associate to has_property
        visisted: all nodes that have been visisted
    '''
    if visited is None:
        visited = set()
    if instance in visited:
        return found
    visited.add(instance)
    for prop in instance.get_properties():
        print('property',prop.name)

        for value in prop[instance]:
                
            print('value',value,type(value))
            print(equals)
            for eq in equals:
                if isinstance(value,Thing) and eq['type'] == 'name' and eq['value'] == value.name:
                    insert = {'type': value.is_a[0].name, 'value': value.name, 'name': prop.name} 
                    if not insert in found:
                        found.append(insert)
                if type(value) == int and eq['type'] == 'value'and eq['value'] > value and eq['op'] == 'less' and eq['name'] == prop.name:
                    insert = {'type': 'int', 'value': value, 'name': prop.name}  
                    if not insert in found:
                        found.append(insert)
                if type(value) == int and eq['type'] == 'value'and eq['value'] < value and eq['op'] == 'greater' and eq['name'] == prop.name:
                    insert = {'type': 'int', 'value': value, 'name': prop.name}  
                    if not insert in found:
                        found.append(insert)
                if type(value) == int and eq['type'] == 'value'and eq['value'] >= value and eq['op'] == 'leq' and eq['name'] == prop.name:
                    found = {'type': 'int', 'value': value, 'name': prop.name} 
                    if not insert in found:
                        found.append(insert)
                if type(value) == int and eq['type'] == 'value'and eq['value'] <= value and eq['op'] == 'geq' and eq['name'] == prop.name:
                    insert = {'type': 'int', 'value': value, 'name': prop.name} 
                    if not insert in found:
                        found.append(insert)
                if type(value) == int and eq['type'] == 'value'and eq['value'] == value and eq['op'] == 'equals' and eq['name'] == prop.name:
                    insert = {'type': 'int', 'value': value, 'name': prop.name}
                    if not insert in found:
                        found.append(insert)
            try: 
                if isinstance(value, Thing): # TODO: this is super redudant and could probably be covered with the above...
                    for has in has_property: 
                        inserts = get_instance_property_values(instance,has)
                        for insert in inserts:
                            insert = {'type': insert.is_a[0].name, 'name': insert.name} 
                            if not insert in found:
                                found.append(insert)
                    find_instance_properties(value, has_property=has_property, found=found,visited=visited,equals=equals)
            except: 
                print('broken')
    return found

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


# def get_instantiated_property_values(instance: Thing, property:) -> list:
#     pass
def get_instance_property_values(instance: Thing, property_name: str) -> list:
    """
    Retrieves the values of a specified property for the given instance.

    Args:
        instance (Thing): The instance for which to retrieve property values.
        property_name (str): The name of the property for which to retrieve values.

    Returns:
        list: A list of values set for the specified property.
    """
    return instance.__getattr__(property_name)

def get_all_property_values_for_instance(instance: Thing) -> dict:
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
        raise ValueError("The instance name must be a string.")
    if not isinstance(onto_class, ThingClass):
        raise ValueError(f"The provided class {onto_class} is not a valid ThingClass.")

    # Create instance with name
    instance = onto_class(instance_name)

    # Assign additional properties if provided
    # TODO: Not sure if this is the correct way to set properties
    try:
        for key, value in properties.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                print(f"Warning: {onto_class.name} has no property '{key}'")

    except Exception as e:
        print(f"Error setting properties for {onto_class.name}: {e}")

    return instance # return instantiated class


def get_instance_class(instance: Thing) -> ThingClass:
    if not isinstance(instance, Thing):
        raise ValueError(f"The provided instance '{instance}' is not a valid Thing.")
    return type(instance)


def assign_object_property_relationship(
    domain: Thing, range: Thing, object_property: ObjectPropertyClass
):
    """
    Connect two Thing instances via a specified ObjectProperty in Owlready2.

    :param domain: The Thing instance representing the domain.
    :param range: The Thing instance representing the range.
    :param object_property: The ObjectProperty to connect the instances.
    """

    # Check if domain and ranges are instances of Thing
    if not isinstance(domain, Thing):
        raise TypeError(f"The 'domain' argument '{domain}' must be an instance of Thing.")
    if not isinstance(range, Thing):
        raise TypeError(f"The 'ranges' argument '{range}' must be of type Thing.")
    if not isinstance(object_property, ObjectPropertyClass):
        raise TypeError(
            f"The 'object_property' argument '{object_property}' must of type ObjectProperty."
        )
    # Connect the two Thing instances
    object_property[domain].append(range)

def link_data_property_to_instance(instance: Thing, data_property: DataPropertyClass, value):
    """
    Links a data property to a Thing instance with a specified value.

    :param instance: The Thing instance to link the data property to.
    :param data_property: The DataProperty to link to the instance.
    :param value: The value to set for the data property.
    """
    try:
        # Check if instance is an instance of Thing
        if not isinstance(instance, Thing):
            raise TypeError("The 'instance' argument must be an instance of Thing.")

        # Check if data_property is a valid DataProperty
        if not isinstance(data_property, DataPropertyClass):
            raise TypeError("The 'data_property' argument must be an instance of DataProperty.")

        # Set the data property value for the instance
        instance.data_property = [value]
    except Exception as e:
        print(f"Error setting data property: {e}")

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

    ontology = get_ontology("data/owl/annett-o-0.1.owl").load()

    # instance1 = create_cls_instance(ontology.Network, "Conv Network")
    # instance2 = create_cls_instance(ontology.CostFunction, "Class1")

    # classes = ontology.CostFunction.is_a
    # print(classes)

    # classes = assign_object_property_relationship(ontology, instance1, instance2)
