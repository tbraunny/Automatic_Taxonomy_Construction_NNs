from owlready2 import ThingClass

def requires_final_instantiation(cls: ThingClass) -> bool:
        """
        Determines if a class requires instantiation as a standalone object.

        A class requires instantiation if has no subclasses.

        :param cls: The class to check.
        :return: True if the class requires final instantiation, else False.
        """
        # has_data_properties = any(cls in prop.domain for prop in ontology.data_properties())
        # has_object_properties = any(cls in prop.domain for prop in ontology.object_properties())
        has_subclasses = any(list(cls.subclasses()))
        return not has_subclasses