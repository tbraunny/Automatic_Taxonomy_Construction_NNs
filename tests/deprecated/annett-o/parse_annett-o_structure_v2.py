from owlready2 import Ontology, ThingClass, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import get_property_range_type


def write_ontology_structure_to_file(ontology: Ontology, file_path: str):
    """
    Writes the ontology's class structure starting from ANNConfiguration,
    ensuring that properties, associated classes, and possible subclasses are listed recursively.
    Atomic values are specified when properties do not link to a class. For previously visited
    classes, a marker indicates the node has been explored before. Classes requiring final
    instantiation are explicitly marked, and subclasses are explored recursively.

    Args:
        ontology (Ontology): the ontology to document.
        file_path (str): the path to the file where the structure will be written.
    """
    def requires_final_instantiation(cls):
        """
        Determines if a class requires final instantiation as a standalone object.

        A class requires final instantiation if it:
        - Has no data properties.
        - Has no object properties.

        Args:
            cls (ThingClass): the class to check.

        Returns:
            bool: true if the class must be instantiated as a final object, otherwise false.
        """
        has_data_properties = any(cls in prop.domain for prop in ontology.data_properties())
        has_object_properties = any(cls in prop.domain for prop in ontology.object_properties())
        return not has_data_properties and not has_object_properties

    def process_subclasses(cls, level, visited_classes):
        """
        recursively processes and writes subclasses of a class.

        args:
            cls (thingclass): the class whose subclasses to process.
            level (int): the current indentation level.
            visited_classes (set): set of already visited classes to avoid duplication.
        """
        subclasses = list(cls.subclasses())  # convert generator to a list, as generators are always truthy
        if not subclasses:  # skip if there are no subclasses
            return

        indent = '    ' * level
        file.write(f"{indent}  Possible Subclasses:\n")
        for subclass in subclasses:
            if subclass in visited_classes:
                file.write(f"{indent}    - {subclass.name} [Already Visited]\n")
            else:
                subclass_instantiation_marker = (
                    " [Final Instantiation Required]" if requires_final_instantiation(subclass) else ""
                )
                file.write(f"{indent}    - {subclass.name}{subclass_instantiation_marker}\n")
                visited_classes.add(subclass)
                process_subclasses(subclass, level + 2, visited_classes)  # subclass of a subclass

    def process_class(cls, level, visited_classes):
        """
        processes a class, writing its properties, recursively processing related classes,
        and listing possible subclasses.

        args:
            cls (thingclass): the class to process.
            level (int): the current indentation level.
            visited_classes (set): set of already visited classes to avoid duplication.
        """
        indent = '    ' * level

        if cls in visited_classes:
            file.write(f"{indent}- Class: {cls.name} [Already Visited]\n")
            return

        visited_classes.add(cls)

        # determine if the class requires final instantiation
        final_instantiation_marker = " [Final Instantiation Required]" if requires_final_instantiation(cls) else ""

        # write class name
        file.write(f"{indent}- Class: {cls.name}{final_instantiation_marker}\n")

        # write data properties
        data_props = [prop for prop in ontology.data_properties() if cls in prop.domain]
        if data_props:
            file.write(f"{indent}  Data Properties:\n")
            for prop in data_props:
                file.write(f"{indent}    - {prop.name} (atomic)\n")

        # write object properties and process their range classes
        obj_props = [prop for prop in ontology.object_properties() if cls in prop.domain]
        if obj_props:
            file.write(f"{indent}  {cls.name} Properties:\n")
            for prop in obj_props:
                range_type = get_property_range_type(prop)

                if range_type == "atomic":
                    file.write(f"{indent}    - {prop.name} (atomic)\n")
                else:
                    file.write(f"{indent}    - {prop.name}\n")
                    for range_cls in prop.range:
                        if isinstance(range_cls, ThingClass):
                            process_class(range_cls, level + 2, visited_classes)

        # recursively process subclasses only if there are any
        if cls.subclasses():
            process_subclasses(cls, level + 1, visited_classes)

    with open(file_path, 'w') as file:
        if not hasattr(ontology, 'ANNConfiguration'):
            print("Class 'ANNConfiguration' does not exist in the ontology.")
            return

        # start processing from ANNConfiguration
        start_class = ontology.ANNConfiguration
        visited_classes = set()
        process_class(start_class, 0, visited_classes)


if __name__ == "__main__":
    file_path = './test.txt'

    # load ontology
    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

    write_ontology_structure_to_file(onto, file_path)
    print("File written successfully.")
