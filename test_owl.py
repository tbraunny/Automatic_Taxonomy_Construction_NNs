from owlready2 import Ontology, ThingClass, Property, Restriction, Thing, DataPropertyClass, ObjectPropertyClass, AnnotationPropertyClass, destroy_entity, sync_reasoner
from typing import List, Dict, Union, Optional
import types
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
    # Assume range of primitive data types (e.g., string, int) is "atomic"
    for range_type in property.range:
        if issubclass(range_type, ThingClass):
            return "class"
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

def get_all_subclasses(onto_class: ThingClass) -> List[ThingClass]:
    """
    Retrieves all direct and indirect subclasses of a specified class.

    Args:
        onto_class (ThingClass): The class for which to find subclasses.

    Returns:
        List[ThingClass]: A list of all subclasses of the specified class.
    """
    return list(onto_class.subclasses())

def get_all_superclasses(onto_class: ThingClass) -> List[ThingClass]:
    """
    Retrieves all direct and indirect superclasses of a specified class.

    Args:
        onto_class (ThingClass): The class for which to find superclasses.

    Returns:
        List[ThingClass]: A list of all superclasses of the specified class.
    """
    return list(onto_class.ancestors())

def get_class_annotations(onto_class: ThingClass) -> Dict[str, List[str]]:
    """
    Retrieves annotations (e.g., labels, comments) associated with a class.

    Args:
        onto_class (ThingClass): The class for which to retrieve annotations.

    Returns:
        Dict[str, List[str]]: A dictionary of annotation properties and their values.
    """
    annotations = {}
    for prop in onto_class.get_annotations():
        annotations[prop.python_name] = getattr(onto_class, prop.python_name)
    return annotations

def create_class(ontology: Ontology, class_name: str, base_classes: Optional[List[ThingClass]] = None) -> ThingClass:
    """
    Dynamically creates a class in the ontology if it does not already exist.

    Args:
        ontology (Ontology): The ontology in which to create the class.
        class_name (str): The name of the class to create.
        base_classes (List[ThingClass], optional): The base classes for the new class. Defaults to [Thing].

    Returns:
        ThingClass: The newly created class or the existing class if it already exists.
    """
    # Check if the class already exists
    existing_class = ontology.search_one(iri="*" + class_name)
    if existing_class:
        warnings.warn(f"Class '{class_name}' already exists.")
        return existing_class

    # Set base classes to Thing if none are provided
    if not base_classes:
        base_classes = [Thing]

    # Dynamically create the new class
    new_class = types.new_class(class_name, tuple(base_classes), kwds={"ontology": ontology})
    return new_class

def delete_class(ontology: Ontology, onto_class: ThingClass):
    """
    Deletes a class from the ontology.

    Args:
        ontology (Ontology): The ontology from which to delete the class.
        onto_class (ThingClass): The class to delete.
    """
    destroy_entity(onto_class)

def get_class_hierarchy(ontology: Ontology) -> Dict[str, List[str]]:
    """
    Constructs a class hierarchy for the ontology.

    Args:
        ontology (Ontology): The ontology for which to construct the hierarchy.

    Returns:
        Dict[str, List[str]]: A dictionary mapping class names to lists of subclass names.
    """
    hierarchy = {}
    for cls in ontology.classes():
        hierarchy[cls.name] = [subcls.name for subcls in cls.subclasses()]
    return hierarchy

def get_classes_using_property(property: Property) -> List[ThingClass]:
    """
    Finds all classes that use a given property.

    Args:
        property (Property): The property to search for.

    Returns:
        List[ThingClass]: A list of classes that use the property.
    """
    classes = []
    for cls in property.domain:
        classes.extend(cls.descendants())
    return list(set(classes))

""" 2) Instance Functions """

def create_instance(onto_class: ThingClass, instance_name: str) -> Thing:
    """
    Creates an instance of a given class.

    Args:
        onto_class (ThingClass): The class of which to create an instance.
        instance_name (str): The name of the new instance.

    Returns:
        Thing: The newly created instance.
    """
    instance = onto_class(instance_name)
    return instance

def delete_instance(instance: Thing):
    """
    Deletes an instance from the ontology.

    Args:
        instance (Thing): The instance to delete.
    """
    destroy_entity(instance)

def get_instances_of_class(onto_class: ThingClass) -> List[Thing]:
    """
    Retrieves all instances of a given class.

    Args:
        onto_class (ThingClass): The class for which to retrieve instances.

    Returns:
        List[Thing]: A list of instances of the class.
    """
    return list(onto_class.instances())

def get_instance_class_properties(instance: Thing) -> List[Property]:
    """
    Retrieves all properties in the ontology that are applicable to the class of a given instance.

    Args:
        instance (Thing): The instance for which to retrieve class-based properties.

    Returns:
        List[Property]: A list of properties that are applicable to the instance's class.
    """
    properties = []
    for cls in instance.is_a:
        properties.extend(get_class_properties(cls.namespace, cls))
    return list(set(properties))

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
        values = prop[instance]
        if values:
            instantiated_properties.append(prop)
    return instantiated_properties

def get_instantiated_property_values(instance: Thing) -> Dict[str, Union[List, str]]:
    """
    Retrieves a dictionary of instantiated properties and their values for the specified instance.

    Args:
        instance (Thing): The instance for which to retrieve property values.

    Returns:
        Dict[str, Union[List, str]]: A dictionary where keys are property names and values are the values set for each property.
    """
    property_values = {}
    for prop in get_instantiated_properties(instance):
        values = prop[instance]
        property_values[prop.name] = values
    return property_values

def get_instance_annotations(instance: Thing) -> Dict[str, List[str]]:
    """
    Retrieves annotations associated with an instance.

    Args:
        instance (Thing): The instance for which to retrieve annotations.

    Returns:
        Dict[str, List[str]]: A dictionary of annotation properties and their values.
    """
    annotations = {}
    for prop in instance.get_annotation_properties():
        annotations[prop.python_name] = getattr(instance, prop.python_name)
    return annotations

def get_instance_relations(instance: Thing) -> Dict[str, List[Thing]]:
    """
    Retrieves relations (object property assertions) for an instance.

    Args:
        instance (Thing): The instance for which to retrieve relations.

    Returns:
        Dict[str, List[Thing]]: A dictionary where keys are property names and values are lists of related instances.
    """
    relations = {}
    for prop in instance.get_properties():
        if isinstance(prop, ObjectPropertyClass):
            values = prop[instance]
            if values:
                relations[prop.name] = values
    return relations

""" 3) Property Functions """

def create_property(ontology: Ontology, property_name: str, property_type: str, domain: List[ThingClass], range: List[Union[ThingClass, type]]) -> Property:
    """
    Creates a new property in the ontology.

    Args:
        ontology (Ontology): The ontology in which to create the property.
        property_name (str): The name of the property to create.
        property_type (str): The type of the property ('ObjectProperty', 'DataProperty', or 'AnnotationProperty').
        domain (List[ThingClass]): The domain of the property.
        range (List[Union[ThingClass, type]]): The range of the property.

    Returns:
        Property: The newly created property.
    """
    if hasattr(ontology, property_name):
        warnings.warn(f"Property '{property_name}' already exists.")
        return getattr(ontology, property_name)

    if property_type == 'ObjectProperty':
        new_property = types.new_class(property_name, (ObjectPropertyClass,), kwds={"ontology": ontology})
    elif property_type == 'DataProperty':
        new_property = types.new_class(property_name, (DataPropertyClass,), kwds={"ontology": ontology})
    elif property_type == 'AnnotationProperty':
        new_property = types.new_class(property_name, (AnnotationPropertyClass,), kwds={"ontology": ontology})
    else:
        raise ValueError("Invalid property_type. Choose 'ObjectProperty', 'DataProperty', or 'AnnotationProperty'.")

    new_property.domain = domain
    new_property.range = range
    return new_property

def delete_property(ontology: Ontology, property: Property):
    """
    Deletes a property from the ontology.

    Args:
        ontology (Ontology): The ontology from which to delete the property.
        property (Property): The property to delete.
    """
    destroy_entity(property)

def get_property_annotations(property: Property) -> Dict[str, List[str]]:
    """
    Retrieves annotations associated with a property.

    Args:
        property (Property): The property for which to retrieve annotations.

    Returns:
        Dict[str, List[str]]: A dictionary of annotation properties and their values.
    """
    annotations = {}
    for prop in property.get_annotation_properties():
        annotations[prop.python_name] = getattr(property, prop.python_name)
    return annotations

def get_property_domain(property: Property) -> List[ThingClass]:
    """
    Retrieves the domain of a property.

    Args:
        property (Property): The property for which to retrieve the domain.

    Returns:
        List[ThingClass]: A list of classes in the domain of the property.
    """
    return list(property.domain)

def get_property_range(property: Property) -> List[Union[ThingClass, type]]:
    """
    Retrieves the range of a property.

    Args:
        property (Property): The property for which to retrieve the range.

    Returns:
        List[Union[ThingClass, type]]: A list of classes or datatypes in the range of the property.
    """
    return list(property.range)

""" 4) Ontology Functions """

def save_ontology(ontology: Ontology, file_path: str, format: str = "rdfxml"):
    """
    Saves the ontology to a file.

    Args:
        ontology (Ontology): The ontology to save.
        file_path (str): The path to the file where the ontology will be saved.
        format (str, optional): The format in which to save the ontology. Defaults to "rdfxml".
    """
    ontology.save(file=file_path, format=format)

def load_ontology(file_path: str) -> Ontology:
    """
    Loads an ontology from a file.

    Args:
        file_path (str): The path to the ontology file.

    Returns:
        Ontology: The loaded ontology.
    """
    from owlready2 import get_ontology
    ontology = get_ontology("file://" + file_path).load()
    return ontology

def list_all_classes(ontology: Ontology) -> List[ThingClass]:
    """
    Lists all classes in the ontology.

    Args:
        ontology (Ontology): The ontology to inspect.

    Returns:
        List[ThingClass]: A list of all classes in the ontology.
    """
    return list(ontology.classes())

def list_all_properties(ontology: Ontology) -> List[Property]:
    """
    Lists all properties in the ontology.

    Args:
        ontology (Ontology): The ontology to inspect.

    Returns:
        List[Property]: A list of all properties in the ontology.
    """
    return list(ontology.properties())

def list_all_instances(ontology: Ontology) -> List[Thing]:
    """
    Lists all instances in the ontology.

    Args:
        ontology (Ontology): The ontology to inspect.

    Returns:
        List[Thing]: A list of all instances in the ontology.
    """
    return list(ontology.individuals())

def infer_ontology(ontology: Ontology):
    """
    Runs a reasoner to infer new knowledge in the ontology.

    Args:
        ontology (Ontology): The ontology on which to run the reasoner.
    """
    with ontology:
        sync_reasoner()

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
            values = prop[instance]
            if values:  # Only print properties that have values
                if isinstance(values, list):
                    values_str = ", ".join(str(v) for v in values)
                else:
                    values_str = str(values)
                print(f"  Property: {prop.name}, Values: {values_str}")
        print("-" * 40)

def list_owl_classes(ontology: Ontology):
    """
    Prints all classes in the ontology.

    Args:
        ontology (Ontology): The ontology to inspect.
    """
    print("Classes in the ontology:")
    for cls in ontology.classes():
        print(f"- {cls.name}")

def list_owl_object_properties(ontology: Ontology):
    """
    Prints all object properties in the ontology.

    Args:
        ontology (Ontology): The ontology to inspect.
    """
    print("\nObject Properties in the ontology:")
    for prop in ontology.object_properties():
        print(f"- {prop.name}")

def list_owl_data_properties(ontology: Ontology):
    """
    Prints all data properties in the ontology.

    Args:
        ontology (Ontology): The ontology to inspect.
    """
    print("\nData Properties in the ontology:")
    for prop in ontology.data_properties():
        print(f"- {prop.name}")

def get_property_class_range(properties: List[Property]) -> Dict[str, List[str]]:
    """
    Retrieves the range (classes) of given properties.

    Args:
        properties (List[Property]): List of properties whose ranges are to be found.

    Returns:
        Dict[str, List[str]]: Dictionary mapping property names to lists of class names in their range.
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


''' Printing '''


def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from ANNConfiguration, including data properties and object properties,
    to a file. It recursively traverses object property ranges as a tree.

    Args:
        ontology (Ontology): The ontology to document.
        file_path (str): The path to the file where the structure will be written.
    """
    with open(file_path, 'w') as file:
        # Start from ANNConfiguration
        if hasattr(ontology, 'ANNConfiguration'):
            visited_classes = set()
            write_class_structure_to_file(ontology.ANNConfiguration, 0, visited_classes, file, ontology)
        else:
            print("Class 'ANNConfiguration' does not exist in the ontology.")

def write_class_structure_to_file(cls: ThingClass, level: int, visited_classes: Set[ThingClass], file, ontology: Ontology):
    """
    Writes the structure of a class to a file, including its data properties and object properties.
    Recursively traverses classes in the range of object properties.

    Args:
        cls (ThingClass): The class to document.
        level (int): The current level in the hierarchy for indentation.
        visited_classes (Set[ThingClass]): Set of already visited classes to prevent infinite loops.
        file: The file object to write to.
        ontology (Ontology): The ontology containing the classes and properties.
    """
    indent = '  ' * level
    if cls in visited_classes:
        return
    visited_classes.add(cls)
    file.write(f"{indent}- Class: {cls.name}\n")

    # Get data properties and object properties
    data_props = get_data_properties_for_class(cls, ontology)
    obj_props = get_object_properties_for_class(cls, ontology)

    # Write data properties to file
    if data_props:
        file.write(f"{indent}  Data Properties:\n")
        for prop in data_props:
            file.write(f"{indent}    - {prop.name}\n")

    # Write object properties to file and recursively traverse range classes
    if obj_props:
        file.write(f"{indent}  Object Properties:\n")
        for prop in obj_props:
            file.write(f"{indent}    - {prop.name}\n")
            for range_cls in prop.range:
                if isinstance(range_cls, ThingClass):
                    write_class_structure_to_file(range_cls, level + 1, visited_classes, file, ontology)

def get_data_properties_for_class(cls: ThingClass, ontology: Ontology) -> Set[DataPropertyClass]:
    """
    Retrieves all data properties where the given class is in the domain.

    Args:
        cls (ThingClass): The class for which to find data properties.
        ontology (Ontology): The ontology containing the properties.

    Returns:
        Set[DataPropertyClass]: A set of data properties with the class in their domain.
    """
    return {prop for prop in ontology.data_properties() if cls in prop.domain}

def get_object_properties_for_class(cls: ThingClass, ontology: Ontology) -> Set[ObjectPropertyClass]:
    """
    Retrieves all object properties where the given class is in the domain.

    Args:
        cls (ThingClass): The class for which to find object properties.
        ontology (Ontology): The ontology containing the properties.

    Returns:
        Set[ObjectPropertyClass]: A set of object properties with the class in their domain.
    """
    return {prop for prop in ontology.object_properties() if cls in prop.domain}


def print_class_structure(onto_class: ThingClass, level=0, visited=None):
    """
    Prints the structure of a specified class, including its subclasses, data properties, and object properties.
    
    Args:
        onto_class (ThingClass): The class whose structure is to be printed.
        level (int, optional): The current level in the hierarchy for indentation.
        visited (set, optional): Set of visited classes to prevent infinite loops.
    """
    if visited is None:
        visited = set()
    indent = '  ' * level
    if onto_class in visited:
        return
    visited.add(onto_class)
    print(f"{indent}- Class: {onto_class.name}")
    
    # Data properties
    data_props = [prop for prop in onto_class.get_class_properties() if isinstance(prop, DataPropertyClass)]
    if data_props:
        print(f"{indent}  Data Properties:")
        for prop in data_props:
            print(f"{indent}    - {prop.name}")
    
    # Object properties
    obj_props = [prop for prop in onto_class.get_class_properties() if isinstance(prop, ObjectPropertyClass)]
    if obj_props:
        print(f"{indent}  Object Properties:")
        for prop in obj_props:
            print(f"{indent}    - {prop.name}")
    
    # Subclasses
    subclasses = onto_class.subclasses()
    for subclass in subclasses:
        print_class_structure(subclass, level + 1, visited)

def print_instance_structure(instance: Thing, level=0, visited=None):
    """
    Prints the structure for a given root of an instantiated class, showing properties and their values.
    
    Args:
        instance (Thing): The instance to inspect.
        level (int, optional): The current level in the hierarchy for indentation.
        visited (set, optional): Set of visited instances to prevent infinite loops.
    """
    if visited is None:
        visited = set()
    indent = '  ' * level
    if instance in visited:
        return
    visited.add(instance)
    print(f"{indent}- Instance: {instance.name}")
    class_names = [cls.name for cls in instance.is_a if cls.name]
    print(f"{indent}  Classes: {', '.join(class_names)}")
    
    # Properties and their values
    for prop in instance.get_properties():
        values = prop[instance]
        if values:
            if isinstance(prop, ObjectPropertyClass):
                print(f"{indent}  Object Property: {prop.name}")
                for value in values:
                    print_instance_structure(value, level + 1, visited)
            elif isinstance(prop, DataPropertyClass):
                values_str = ', '.join(str(v) for v in values)
                print(f"{indent}  Data Property: {prop.name}, Value: {values_str}")

def print_instances_of_class(onto_class: ThingClass):
    """
    Prints all instances of a given class.
    
    Args:
        onto_class (ThingClass): The class whose instances are to be printed.

    Example input:
        onto.TaskCharacterization
    """
    instances = list(onto_class.instances())
    if instances:
        print(f"Instances of class '{onto_class.name}':")
        for instance in instances:
            print(f"- {instance.name}")
    else:
        print(f"No instances found for class '{onto_class.name}'.")


def print_instances_of_data_property(data_property: DataPropertyClass):
    """
    Prints all instances (values) of a given data property.

    Args:
        data_property (DataPropertyClass): The data property whose instances (values) are to be printed.

    Example input:
        onto.layer_num_units
    """
    instances = data_property.get_relations()
    if instances:
        print(f"Values for data property '{data_property.name}':")
        for subject, value in instances:
            print(f"- Instance: {subject.name} | Value: {value}")
    else:
        print(f"No values found for data property '{data_property.name}'.")

