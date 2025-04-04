import unittest
from owlready2 import (get_ontology, ThingClass, ObjectPropertyClass, DataPropertyClass, Thing)

from utils.owl_utils import (
    get_highest_subclass_ancestor,
    get_class_parents,
    get_domain_class,
    get_onto_object_from_str,
    is_subclass_of_class,
    is_subclass_of_any,
    get_object_properties_with_domain_and_range,
    get_connected_classes,
    get_immediate_subclasses,
    get_all_subclasses,
    create_class,
    create_subclass,
    get_class_instances,
    create_cls_instance,
    list_owl_classes,
    list_owl_object_properties,
    list_owl_data_properties,
)

class TestOwlUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test ontology
        cls.ontology = get_ontology("http://test.org/onto.owl")
        with cls.ontology:
            class BaseClass(Thing):
                pass

            class SubClass1(BaseClass):
                pass

            class SubClass2(BaseClass):
                pass

            class TestObjectProperty(ObjectPropertyClass):
                domain = [BaseClass]
                range = [SubClass1]

            class TestDataProperty(DataPropertyClass):
                domain = [BaseClass]
                range = [str]

    def test_get_highest_subclass_ancestor(self):
        subclass = self.ontology.SubClass1
        result = get_highest_subclass_ancestor(subclass)
        self.assertEqual(result, self.ontology.BaseClass)

    def test_get_class_parents(self):
        subclass = self.ontology.SubClass1
        result = get_class_parents(subclass)
        self.assertIn(self.ontology.BaseClass, result)

    def test_get_domain_class(self):
        obj_property = self.ontology.TestObjectProperty
        result = get_domain_class(self.ontology, obj_property)
        self.assertIn(self.ontology.BaseClass, result)

    def test_get_onto_object_from_str(self):
        result = get_onto_object_from_str(self.ontology, "SubClass1")
        self.assertEqual(result, self.ontology.SubClass1)

    def test_is_subclass_of_class(self):
        result = is_subclass_of_class(self.ontology.SubClass1, self.ontology.BaseClass)
        self.assertTrue(result)

    def test_is_subclass_of_any(self):
        result = is_subclass_of_any(self.ontology.SubClass1)
        self.assertTrue(result)

    def test_get_object_properties_with_domain_and_range(self):
        result = get_object_properties_with_domain_and_range(
            self.ontology, self.ontology.BaseClass, self.ontology.SubClass1
        )
        self.assertEqual(result, self.ontology.TestObjectProperty)

    def test_get_connected_classes(self):
        result = get_connected_classes(self.ontology.BaseClass, self.ontology)
        self.assertIn(self.ontology.SubClass1, result)

    def test_get_immediate_subclasses(self):
        result = get_immediate_subclasses(self.ontology.BaseClass)
        self.assertIn(self.ontology.SubClass1, result)
        self.assertIn(self.ontology.SubClass2, result)

    def test_get_all_subclasses(self):
        result = get_all_subclasses(self.ontology.BaseClass)
        self.assertIn(self.ontology.SubClass1, result)
        self.assertIn(self.ontology.SubClass2, result)

    def test_create_class(self):
        new_class = create_class(self.ontology, "NewClass", self.ontology.BaseClass)
        self.assertTrue(hasattr(self.ontology, "NewClass"))
        self.assertTrue(issubclass(new_class, self.ontology.BaseClass))

    def test_create_subclass(self):
        new_subclass = create_subclass(
            self.ontology, "NewSubClass", self.ontology.BaseClass
        )
        self.assertTrue(hasattr(self.ontology, "NewSubClass"))
        self.assertTrue(issubclass(new_subclass, self.ontology.BaseClass))

    def test_get_class_instances(self):
        instance = self.ontology.SubClass1("instance1")
        result = get_class_instances(self.ontology.SubClass1)
        self.assertIn(instance, result)

    def test_create_cls_instance(self):
        instance = create_cls_instance(
            self.ontology, self.ontology.SubClass1, "instance2"
        )
        self.assertEqual(instance.name, "instance2")
        self.assertIn(instance, self.ontology.SubClass1.instances())

    def test_list_owl_classes(self):
        result = list_owl_classes(self.ontology)
        self.assertIn(self.ontology.BaseClass, result)
        self.assertIn(self.ontology.SubClass1, result)

    def test_list_owl_object_properties(self):
        result = list_owl_object_properties(self.ontology)
        self.assertIn(self.ontology.TestObjectProperty, result)

    def test_list_owl_data_properties(self):
        result = list_owl_data_properties(self.ontology)
        print(f"Resilts: {result}")
        self.assertIn(self.ontology.TestDataProperty, result)


if __name__ == "__main__":
    unittest.main()