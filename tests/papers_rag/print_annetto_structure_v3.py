from owlready2 import Ontology, ThingClass, get_ontology
from utils.constants import Constants as C
from utils.owl.owl import get_property_range_type

def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from ANNConfiguration.
    It details properties, associated classes, and subclasses recursively.
    
    :param ontology: The ontology object to analyze.
    :param file_path: The path to the output file.
    """
    
    def requires_final_instantiation(cls: ThingClass) -> bool:
        """
        Determines if a class requires instantiation as a standalone object.

        A class requires instantiation if it has neither data nor object properties.
        
        :param cls: The class to check.
        :return: True if the class requires final instantiation, else False.
        """
        has_data_properties = any(cls in prop.domain for prop in ontology.data_properties())
        has_object_properties = any(cls in prop.domain for prop in ontology.object_properties())
        has_sub_classes = any(list(cls.subclasses()))
        return (not has_data_properties and not has_object_properties) and not has_sub_classes
    
    def process_subclasses(cls: ThingClass, level: int, visited_classes: set):
        """
        Recursively processes and writes subclasses of a given class.

        :param cls: The parent class whose subclasses are processed.
        :param level: The current indentation level in the output file.
        :param visited_classes: A set of previously visited classes to avoid duplication.
        """
        subclasses = list(cls.subclasses())  # Convert generator to list to check emptiness
        if not subclasses:
            return
        
        indent = '    ' * level
        file.write(f"{indent}  Possible Subclasses:\n")
        
        for subclass in subclasses:
            if subclass in visited_classes:
                file.write(f"{indent}    - {subclass.name} [Already Visited]\n")
            else:
                subclass_marker = " [Final Instantiation Required]" if requires_final_instantiation(subclass) else ""
                file.write(f"{indent}    - {subclass.name}{subclass_marker}\n")
                visited_classes.add(subclass)
                process_subclasses(subclass, level + 2, visited_classes)
    
    def process_class(cls: ThingClass, level: int, visited_classes: set):
        """
        Processes a class, writing its properties and recursively handling related classes.

        :param cls: The class to process.
        :param level: The indentation level for readability.
        :param visited_classes: A set to track visited classes.
        """
        indent = '    ' * level

        if cls in visited_classes:
            file.write(f"{indent}- Class: {cls.name} [Already Visited]\n")
            return
        
        visited_classes.add(cls)
        final_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""
        file.write(f"{indent}- Class: {cls.name}{final_marker}\n")

        # Process Data Properties
        data_properties = [prop for prop in ontology.data_properties() if cls in prop.domain]
        if data_properties:
            file.write(f"{indent}  Data Properties:\n")
            for prop in data_properties:
                file.write(f"{indent}    - {prop.name} (atomic)\n")

        # Process Object Properties
        object_properties = [prop for prop in ontology.object_properties() if cls in prop.domain]
        if object_properties:
            file.write(f"{indent}  {cls.name} Properties:\n")
            for prop in object_properties:
                range_type = get_property_range_type(prop)
                
                if range_type == "atomic":
                    file.write(f"{indent}    - {prop.name} (atomic)\n")
                else:
                    file.write(f"{indent}    - {prop.name}\n")
                    for range_cls in prop.range:
                        if isinstance(range_cls, ThingClass):
                            process_class(range_cls, level + 2, visited_classes)

        # Recursively Process Subclasses
        if cls.subclasses():
            process_subclasses(cls, level + 1, visited_classes)
    
    with open(file_path, 'w') as file:
        # Ensure ANNConfiguration exists in the ontology
        if not hasattr(ontology, 'ANNConfiguration'):
            print("Error: Class 'ANNConfiguration' not found in ontology.")
            return

        start_class = ontology.ANNConfiguration
        visited_classes = set()
        process_class(start_class, 0, visited_classes)
        print("Ontology structure written successfully.")


if __name__ == "__main__":
    """
    Main execution block to load the ontology and write its structure to a file.
    """
    # Define output file path
    OUTPUT_FILE = './test.txt'

    # Load ontology
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    onto = get_ontology(ontology_path).load()

    # Write ontology structure to file
    write_ontology_structure_to_file(onto, OUTPUT_FILE)
