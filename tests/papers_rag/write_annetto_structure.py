from owlready2 import Ontology, ThingClass, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties, get_connected_classes, get_subclasses
)
from utils.annetto_utils import requires_final_instantiation, subclasses_requires_final_instantiation

OMIT_CLASSES = ["DataCharacterization", "Regularization"]

def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from selected classes.
    Uses a single global set (processed_classes) to prevent cycles across both
    subclass and object-property (connected) recursion.
    """
    with open(file_path, 'w') as file:

        def _get_final_marker(cls: ThingClass) -> str:
            """Return marker flags based on instantiation requirements."""
            marker = ""
            if requires_final_instantiation(cls):
                marker += " [FIR]"
            if subclasses_requires_final_instantiation(cls):
                marker += "<IS>"
            return marker

        def _write_data_properties(indent: str, cls: ThingClass):
            """Write all data properties of a class, if any."""
            props = get_class_data_properties(ontology, cls)
            if props:
                for prop in props:
                    file.write(f"{indent}        - Data Prop: {prop.name} (atomic)\n")

        def _process_entity(cls: ThingClass, label: str, level: int, processed_classes: set):
            """
            Process an entity (class, connected class, or subclass) by writing its
            header, its data properties, and then recursively processing its connected
            classes and subclasses.
            """
            indent = "    " * level

            # Skip if already processed or omitted.
            if cls in processed_classes or cls.name in OMIT_CLASSES:
                return

            processed_classes.add(cls)
            file.write(f"{indent}- {label}: {cls.name}{_get_final_marker(cls)}\n")
            _write_data_properties(indent, cls)

            # Process connected classes via object properties.
            connected = get_connected_classes(cls, ontology)
            if connected:
                for conn in connected:
                    if isinstance(conn, ThingClass):
                        _process_entity(conn, "Connected Class", level + 2, processed_classes)
                    else:
                        file.write(f"{indent}        - {conn} (non-class connection?)\n")

            # Process subclasses.
            subs = get_subclasses(cls)
            if subs:
                for sub in subs:
                    _process_entity(sub, "Subclass", level + 2, processed_classes)

        # Check for the required key class.
        if not hasattr(ontology, 'ANNConfiguration'):
            print("Error: Class 'ANNConfiguration' not found in ontology.")
            return

        processed_classes = set()

        # Process the top-level classes.
        if hasattr(ontology, "Network"):
            _process_entity(ontology.Network, "Class", 0, processed_classes)
        if hasattr(ontology, "TrainingStrategy"):
            _process_entity(ontology.TrainingStrategy, "Class", 0, processed_classes)

        print("Ontology structure written successfully.")

if __name__ == "__main__":
    OUTPUT_FILE = './annetto_structure_test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()    

    write_ontology_structure_to_file(ontology, OUTPUT_FILE)