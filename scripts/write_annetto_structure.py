from owlready2 import Ontology, ThingClass, Thing
from utils.owl_utils import (
    get_class_data_properties, get_connected_classes, get_immediate_subclasses, get_class_instances, get_class_object_properties
)
from utils.annetto_utils import load_annetto_ontology
from typing import Union

OMIT_CLASSES = ["DataCharacterization", "Regularization"]
OMIT_CLASSES = []


def write_ontology_structure_to_file(root: Union[ThingClass, Thing], ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from selected classes.
    Uses a single global set (processed_classes) to prevent cycles across both
    subclass and object-property (connected) recursion.
    """
    with open(file_path, 'w') as file:

        def _write_data_properties(indent: str, cls: Union[ThingClass, Thing]):
            """Write all data properties of a class, if any."""
            props = get_class_data_properties(ontology, cls)
            if props:
                for prop in props:
                    if prop is not None:  # Ensure prop is not None
                        file.write(f"{indent}        - Data Prop: {prop.name} (atomic)\n")

        def _process_entity(cls: Union[ThingClass, Thing], label: str, level: int, processed_classes: set, root_type: type):
            """
            Process an entity (class, connected class, or subclass) by writing its
            header, its data properties, and then recursively processing its connected
            classes and subclasses.
            """
            indent = "    " * level

            # Skip if already processed or omitted.
            if cls in processed_classes or cls.name in OMIT_CLASSES:
                return
            
            obj_prop = [prop if cls in prop.range else None for prop in ontology.object_properties()]
            


            processed_classes.add(cls)
            if label == "Connected Class":
                file.write(f"{indent}- {label}: {cls.name} on prop {[p for p in obj_prop if p]}\n")
            else:
                file.write(f"{indent}- {label}: {cls.name}\n")
            
            _write_data_properties(indent, cls)

            # Process connected classes via object properties.
            connected = get_connected_classes(cls, ontology)
            if connected:
                for conn in connected:
                    if isinstance(conn, root_type):  # Ensure the type matches the root type
                        _process_entity(conn, "Connected Class", level + 2, processed_classes, root_type)
                    else:
                        file.write(f"{indent}        - {conn} (non-class connection?)\n")

            # Process subclasses if the root type is ThingClass.
            if root_type is ThingClass and isinstance(cls, ThingClass):
                subs = get_immediate_subclasses(cls)
                if subs:
                    for sub in subs:
                        _process_entity(sub, "Subclass", level + 2, processed_classes, root_type)

        # Determine the type of the root entity.
        root_type = ThingClass if isinstance(root, ThingClass) else Thing

        # Check for the required key class.
        if not hasattr(ontology, 'ANNConfiguration'):
            print("Error: Class 'ANNConfiguration' not found in ontology.")
            return

        processed_classes = set()

        if root is ontology.ANNConfiguration:
            _process_entity(ontology.Network, "Class", 0, processed_classes, root_type)
            _process_entity(ontology.TrainingStrategy, "Class", 0, processed_classes, root_type)
        else:
            _process_entity(root, "Class", 0, processed_classes, root_type)

        print("Ontology structure written successfully.")

if __name__ == "__main__":
    OUTPUT_FILE = './annetto_structure_test.txt'
    ontology = load_annetto_ontology(release_type="test")
    print("Loaded ontology:", ontology)
    # Process the top-level classes.
    list_instances = get_class_instances(ontology.ANNConfiguration)
    instance = list_instances[3] if list_instances else None
    root = instance

    # write_ontology_structure_to_file(root, ontology, OUTPUT_FILE)
    write_ontology_structure_to_file(ontology.ANNConfiguration, ontology, OUTPUT_FILE)
