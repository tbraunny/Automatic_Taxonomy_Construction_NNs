from owlready2 import Ontology, ThingClass, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import get_class_data_properties, get_connected_classes, get_subclasses
from utils.annetto_utils import requires_final_instantiation

OMIT_CLASSES = ["DataCharacterization"]

def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from selected classes.
    This version uses a single global set (processed_classes) to prevent cycles
    across both subclass and object-property (connected) recursion.
    """

    def process_connected(cls: ThingClass, level: int, processed_classes: set):
        indent = "    " * level

        # If we've processed this class already, note it and return.
        if cls in processed_classes or cls.name in OMIT_CLASSES:
            # file.write(f"{indent}- Connected Class: {cls.name} (already processed)\n")
            return

        processed_classes.add(cls)
        final_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""
        file.write(f"{indent}- Connected Class: {cls.name}{final_marker}\n")

        # Data properties
        data_properties = get_class_data_properties(ontology, cls)
        if data_properties:
            file.write(f"{indent}    Data Properties:\n")
            for prop in data_properties:
                file.write(f"{indent}        - {prop.name} (atomic)\n")

        # Process further object property connections
        connected = get_connected_classes(cls, ontology)
        if connected:
            # file.write(f"{indent}    Connected Classes (via Obj Properties):\n")
            for conn in connected:
                if isinstance(conn, ThingClass):
                    process_connected(conn, level + 2, processed_classes)
                else:
                    file.write(f"{indent}        - {conn} (non-class connection?)\n")

        # Optionally, if you want to see the subclass structure for a connected class,
        # you can process its subclasses. (If not, comment out the following block.)
        subs = get_subclasses(cls)
        if subs:
            file.write(f"{indent}    Subclasses of {cls}:\n")
            for sub in subs:
                process_subclass(sub, level + 2, processed_classes)

    def process_subclass(subclass: ThingClass, level: int, processed_classes: set):
        indent = "    " * level

        if subclass in processed_classes or subclass.name in OMIT_CLASSES:
            # file.write(f"{indent}- Subclass: {subclass.name} (already processed or omitted)\n")
            return

        processed_classes.add(subclass)
        final_marker = " [Final Instantiation Required]" if requires_final_instantiation(subclass) else ""
        file.write(f"{indent}- Subclass: {subclass.name}{final_marker}\n")

        # Data properties for the subclass
        data_properties = get_class_data_properties(ontology, subclass)
        if data_properties:
            file.write(f"{indent}    Data Properties:\n")
            for prop in data_properties:
                file.write(f"{indent}        - {prop.name} (atomic)\n")

        # Process object property connections from the subclass
        connected_classes = get_connected_classes(subclass, ontology)
        if connected_classes:
            # file.write(f"{indent}    Connected Classes (via Obj Properties):\n")
            for connected_class in connected_classes:
                if isinstance(connected_class, ThingClass):
                    process_connected(connected_class, level + 2, processed_classes)
                else:
                    file.write(f"{indent}        - {connected_class} (non-class connection?)\n")

        # Process deeper subclasses
        deeper_subclasses = get_subclasses(subclass)
        if deeper_subclasses:
            for deeper_subclass in deeper_subclasses:
                process_subclass(deeper_subclass, level + 2, processed_classes)

    def process_class(cls: ThingClass, level: int, processed_classes: set):
        indent = "    " * level

        if cls in processed_classes or cls.name in OMIT_CLASSES:
            # file.write(f"{indent}- Class: {cls.name} (already processed or omitted)\n")
            return

        processed_classes.add(cls)
        final_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""
        file.write(f"{indent}- Class: {cls.name}{final_marker}\n")

        # Data Properties
        data_properties = get_class_data_properties(ontology, cls)
        if data_properties:
            file.write(f"{indent}    Data Properties:\n")
            for prop in data_properties:
                file.write(f"{indent}        - {prop.name} (atomic)\n")

        # Process object property connections (using the same global processed set)
        connected_classes = get_connected_classes(cls, ontology)
        if connected_classes:
            # file.write(f"{indent}    Connected Classes (via Obj Properties):\n")
            for connected_class in connected_classes:
                if isinstance(connected_class, ThingClass):
                    process_connected(connected_class, level + 2, processed_classes)
                else:
                    file.write(f"{indent}        - {connected_class} (non-class connection?)\n")

        # Process subclasses
        subclasses = get_subclasses(cls)
        if subclasses:
            for subcls in subclasses:
                process_subclass(subcls, level + 2, processed_classes)

    with open(file_path, 'w') as file:
        # Ensure that key classes exist.
        if not hasattr(ontology, 'ANNConfiguration'):
            print("Error: Class 'ANNConfiguration' not found in ontology.")
            return

        processed_classes = set()

        # Process selected top-level classes.
        if hasattr(ontology, "Network"):
            process_class(ontology.Network, 0, processed_classes)
        if hasattr(ontology, "TrainingStrategy"):
            process_class(ontology.TrainingStrategy, 0, processed_classes)

        print("Ontology structure written successfully.")

if __name__ == "__main__":
    OUTPUT_FILE = './annetto_structure_test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    onto = get_ontology(ontology_path).load()
    write_ontology_structure_to_file(onto, OUTPUT_FILE)

# from owlready2 import Ontology, ThingClass, get_ontology
# from utils.constants import Constants as C
# from utils.owl_utils import get_class_data_properties, get_connected_classes, get_subclasses
# from utils.annetto_utils import requires_final_instantiation

# OMIT_CLASSES = ["DataCharacterization"]


# def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
#     """
#     Writes the ontology's class structure starting from selected classes.
#     Demonstrates how to keep a global visited set and a local visited_subclasses set.
#     """

#     def process_subclass(subclass: ThingClass, level: int, visited_classes:set, visited_subclasses: set):
#         """
#         Recursively process a subclass, using a 'visited_subclasses' set
#         scoped to this local branch only.
#         """

#         if subclass in visited_classes:
#             return
        
#         if subclass in visited_subclasses:
#             # Already encountered this exact subclass in the same branch
#             return
        
#         if subclass.name in OMIT_CLASSES:
#             return
        
#         visited_subclasses.add(subclass)

#         indent = "    " * level
#         final_marker = " [Final Instantiation Required]" if requires_final_instantiation(subclass) else ""
#         file.write(f"{indent}- Subclass: {subclass.name}{final_marker}\n")

#         # If you want to show data properties for the subclass as well:
#         data_properties = get_class_data_properties(ontology, subclass)
#         if data_properties:
#             file.write(f"{indent}    Data Properties:\n")
#             for prop in data_properties:
#                 file.write(f"{indent}        - {prop.name} (atomic)\n")

#         # --- Object Property Connections ---
#         connected_classes = get_connected_classes(subclass, ontology)
#         if connected_classes:
#             file.write(f"{indent}    Connected Classes (via Obj Properties):\n")
#             for connected_class in connected_classes:
#                 if isinstance(connected_class, ThingClass):
#                     process_subclass(connected_class, level + 2, visited_classes,visited_subclasses)
#                 else:
#                     file.write(f"{indent}        - {connected_class} (non-class connection?)\n")

#         # Recursively process deeper subclasses of this subclass
#         deeper_subclasses = get_subclasses(subclass)
#         if deeper_subclasses:
#             # file.write(f"{indent}    Subclasses of {subclass.name}:\n")
#             for deeper_subclass in deeper_subclasses:
#                 if deeper_subclass in visited_classes:
#                     continue
#                 process_subclass(deeper_subclass, level + 2, visited_classes, visited_subclasses)

#     def process_class(cls: ThingClass, level: int, visited_classes: set):
#         """
#         Processes a class, preventing re-processing via 'visited_classes'.
#         Then uses a *new* local set for subclasses so repeated references 
#         only block within one branch, not the entire traversal.
#         """
#         # Skip if we've processed this class globally (via object properties or top-level calls)
#         if cls in visited_classes:
#             return
        
#         if cls.name in OMIT_CLASSES:
#             return
#         visited_classes.add(cls)

#         indent = "    " * level
#         final_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""
#         file.write(f"{indent}- Class: {cls.name}{final_marker}\n")

#         # --- Data Properties ---
#         data_properties = get_class_data_properties(ontology, cls)
#         if data_properties:
#             file.write(f"{indent}    Data Properties:\n")
#             for prop in data_properties:
#                 file.write(f"{indent}        - {prop.name} (atomic)\n")

#          # --- Object Property Connections ---
#         connected_classes = get_connected_classes(cls, ontology)

#         if connected_classes:
#             file.write(f"{indent}    Connected Classes (via Obj Properties):\n")
#             for connected_class in connected_classes:
#                 if isinstance(connected_class, ThingClass):
#                     process_class(connected_class, level + 2, visited_classes)
#                 else:
#                     file.write(f"{indent}        - {connected_class} (non-class connection?)##############\n")

#         # --- Subclasses ---
#         subclasses = get_subclasses(cls)
#         if subclasses:
#             # Pass a *fresh* local set of visited_subclasses 
#             # so each top-level class has its own local recursion scope
#             local_visited_subclasses = set()
#             for subcls in subclasses:
#                 process_subclass(subcls, level + 2, visited_classes, local_visited_subclasses)

       

#     with open(file_path, 'w') as file:
#         # Check that ANNConfiguration or other key classes exist
#         if not hasattr(ontology, 'ANNConfiguration'):
#             print("Error: Class 'ANNConfiguration' not found in ontology.")
#             return

#         visited_classes = set()

#         # Process Network first and TrainingStrategy second
#         if hasattr(ontology, "Network"):
#             process_class(ontology.Network, 0, visited_classes)
#         if hasattr(ontology, "TrainingStrategy"):
#             process_class(ontology.TrainingStrategy, 0, visited_classes)

#         print("Ontology structure written successfully.")


# if __name__ == "__main__":
#     # Define output file path
#     OUTPUT_FILE = './annetto_structure_test.txt'

#     # Load ontology
#     ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
#     onto = get_ontology(ontology_path).load()

#     # Write ontology structure to file
#     write_ontology_structure_to_file(onto, OUTPUT_FILE)








# from owlready2 import Ontology, ThingClass, get_ontology
# from utils.constants import Constants as C
# from utils.owl_utils import get_class_data_properties, get_connected_classes
# from utils.annetto_utils import requires_final_instantiation


# def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
#     """
#     Writes the ontology's class structure in a hierarchical manner.

#     :param ontology: The ontology object to analyze.
#     :param file_path: The path to the output file.
#     """

#     # def process_subclasses(subclass: ThingClass, level: int, visited_subclasses: set = None):
#     #     """
#     #     Recursively processes and writes a subclass of a given class.

#     #     :param cls: The subclass to be processed.
#     #     :param level: The current indentation level in the output file.
#     #     :param visited_classes: A set of previously visited subclasses to avoid duplication.
#     #     """

#     #     if visited_subclasses is None:  # Initialize set
#     #         visited_subclasses = set()

#     #     indent = '    ' * level

#     #     subclass_marker = " [Final Instantiation Required]" if requires_final_instantiation(subclass) else ""

#     #     if subclass in visited_subclasses:
#     #         subclass_marker + "Already seen"
#     #         return

#     #     file.write(f"{indent}    - {subclass.name}{subclass_marker}\n")
#     #     visited_subclasses.add(subclass)

#     #     if subclass.subclasses():
#     #         file.write(f"{indent} Subclasses:\n")
#     #         for subsubclass in subclass.subclasses():
#     #             process_subclasses(subsubclass, level + 2, visited_subclasses)

#     def process_class(cls: ThingClass, level: int, visited_classes: set, visited_subclasses: set = None):
#         """
#         Recursively processes a class, writing its properties, subclasses,
#         and object-property connections.

#         :param cls: The class to process.
#         :param level: The indentation level for readability.
#         :param visited_classes: A set to track visited classes (avoid loops).
#         """
#         if cls in visited_classes:
#             return
        
#         if cls in ["DataCharacterization"]:
#             return
        
#         visited_classes.add(cls)

#         indent = "    " * level
#         # Add marker if final instantiation is required:
#         final_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""
#         file.write(f"{indent}- Class: {cls.name}{final_marker}\n")

#         # --- Data Properties ---
#         data_properties = get_class_data_properties(ontology, cls)
#         if data_properties:
#             file.write(f"{indent}    Data Properties:\n")
#             for prop in data_properties:
#                 file.write(f"{indent}        - {prop.name} (atomic)\n")

#         # --- Subclasses ---
#         subclasses = cls.subclasses()
#         if subclasses:
#             file.write(f"{indent}    Subclasses:\n")
#             for subcls in subclasses:
#                 process_class(subcls, level + 2, visited_classes)

#         # --- Object Property Connections ---
#         connected_classes = get_connected_classes(cls, ontology)
#         if connected_classes:
#             file.write(f"{indent}    Connected Classes (via Obj Properties):\n")
#             for connected_class in connected_classes:
#                 if isinstance(connected_class, ThingClass):
#                     process_class(connected_class, level + 2, visited_classes)
#                 else:
#                     # If you want to note non-class connections:
#                     file.write(f"{indent}        - {connected_class}\n")

#     with open(file_path, 'w') as file:
#         # Ensure ANNConfiguration or other key classes exist in the ontology
#         if not hasattr(ontology, 'ANNConfiguration'):
#             print("Error: Class 'ANNConfiguration' not found in ontology.")
#             return
        
#         visited_classes = set()

#         # If you specifically want to process Network and TrainingStrategy first:
#         if hasattr(ontology, "Network"):
#             process_class(ontology.Network, 0, visited_classes)

#         if hasattr(ontology, "TrainingStrategy"):
#             process_class(ontology.TrainingStrategy, 0, visited_classes)

#         print("Ontology structure written successfully.")


# if __name__ == "__main__":
#     # Define output file path
#     OUTPUT_FILE = './annetto_structure_test.txt'

#     # Load ontology
#     ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
#     onto = get_ontology(ontology_path).load()

#     # Write ontology structure to file
#     write_ontology_structure_to_file(onto, OUTPUT_FILE)












# from owlready2 import Ontology, ThingClass, get_ontology
# from utils.constants import Constants as C
# from utils.owl_utils import get_class_data_properties, get_connected_classes, get_subclasses
# from utils.annetto_utils import requires_final_instantiation


# def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
#     """
#     Writes the ontology's class structure starting from ANNConfiguration.
#     Recursively details properties, associated classes, and subclasses.
    
#     :param ontology: The ontology object to analyze.
#     :param file_path: The path to the output file.
#     """
    
#     def process_subclasses(subclass: ThingClass, level: int, visited_subclasses: set = None):
#         """
#         Recursively processes and writes a subclass of a given class.

#         :param cls: The subclass to be processed.
#         :param level: The current indentation level in the output file.
#         :param visited_classes: A set of previously visited subclasses to avoid duplication.
#         """

#         if visited_subclasses is None:  # Initialize set
#             visited_subclasses = set()

#         # subclasses = get_subclasses(cls)
#         # if not subclasses:
#         #     return
        
#         indent = '    ' * level
#         # file.write(f"{indent} Subclasses:\n")
        
#         # for subclass in subclasses:
#             # if subclass in visited_classes:
#             #     continue
            
#         subclass_marker = " [Final Instantiation Required]" if requires_final_instantiation(subclass) else ""

#         if subclass in visited_subclasses:
#             subclass_marker + "Already seen"
#             return

#         file.write(f"{indent}    - {subclass.name}{subclass_marker}\n")
#         visited_subclasses.add(subclass)
#         # visited_classes.add(subclass)
#         # process_subclasses(subclass, level + 2, visited_subclasses)
#         if subclass.subclasses():
#             file.write(f"{indent} Subclasses:\n")
#             for subsubclass in subclass.subclasses():
#                 process_subclasses(subsubclass, level + 2, visited_subclasses)
    
#     def process_class(cls: ThingClass, level: int, visited_classes: set):
#         """
#         Processes a class, writing its properties and recursively handling related classes.

#         :param cls: The class to process.
#         :param level: The indentation level for readability.
#         :param visited_classes: A set to track visited classes.
#         """
#         indent = '    ' * level

#         if cls in visited_classes:
#             return
        
#         # if cls.name in ["DataCharacterization"]:
#         #     return
        
#         visited_classes.add(cls)

#         indent = "    " * level
#         final_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""
#         file.write(f"{indent}- Class: {cls.name}{final_marker}\n")
        
#         # Process Data Properties
#         data_properties = get_class_data_properties(ontology, cls)
#         if data_properties:
#             file.write(f"{indent}  Data Properties:\n")
#             for prop in data_properties:
#                 file.write(f"{indent}    - {prop.name} (atomic)\n")


#         # Recursively Process Subclasses
#         if cls.subclasses():
#             # indent = '    ' * level
#             file.write(f"{indent}  {cls.name} Subclasses:\n")

#             for subclass in cls.subclasses():
#                 process_subclasses(subclass, level + 1)
        
#         # Recursively Process Related Classes

#         connected_classes_by_object_property = get_connected_classes(cls, ontology)
#         if connected_classes_by_object_property:
#             file.write(f"{indent}  {cls.name} Obj Properties:\n")
#             for connected_class in connected_classes_by_object_property:
#                 if isinstance(connected_class, ThingClass):
#                     process_class(connected_class, level + 2, visited_classes)
#                 # else:
#                 #     file.write(f"{indent}    - {connected_class.name} (atomic)\n")
#                 #     print(f"possible data property??? {connected_class}")


#     with open(file_path, 'w') as file:
#         # Ensure ANNConfiguration exists in the ontology
#         if not hasattr(ontology, 'ANNConfiguration'):
#             print("Error: Class 'ANNConfiguration' not found in ontology.")
#             return

#         start_class = ontology.ANNConfiguration
#         visited_classes = set()

#         # Process Netork Class Before Training Strategy
#         # Uses the same visited classes set so that no cylics are made
#         process_class(ontology.Network, 0, visited_classes)
#         process_class(ontology.TrainingStrategy, 0, visited_classes)

#         print("Ontology structure written successfully.")


# if __name__ == "__main__":
#     """
#     Main execution block to load the ontology and write its structure to a file.
#     """
#     # Define output file path
#     OUTPUT_FILE = './annetto_structure_test.txt'

#     # Load ontology
#     ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
#     onto = get_ontology(ontology_path).load()

#     # Write ontology structure to file
#     write_ontology_structure_to_file(onto, OUTPUT_FILE)
