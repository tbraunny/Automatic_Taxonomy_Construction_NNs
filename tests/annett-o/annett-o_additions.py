from owlready2 import *
from utils.constants import Constants as C
from utils.owl_utils import *

from utils.annetto_utils import load_annetto_ontology


def add_optimized_estimators(onto: Ontology):  # TODO: Function not fully implemented.
    """
    Adds the OptimizedEstimators class to the ontology
    as a disjoint sibling of the LossFunction class.

    This is meant to account for cost functions that are meant to maximize (i.e. Maximizing Likelihood) rather
    than minimize the like loss function.

    :param: onto: The ontology to which the OptimizedEstimators class will be added.
    """

    loss_function_class = onto.LossFunction

    if loss_function_class:
        # Extract characteristics of Loss Function
        print("Loss Function Characteristics:")
        print(f"Direct Subclasses: {get_immediate_subclasses(loss_function_class)}")
        print(f"Indirect Subclasses: {get_all_subclasses(loss_function_class)}")
        print(f"Superclasses: {[sup.name for sup in loss_function_class.is_a]}")
        print(
            f"Object Properties: {get_object_properties_for_class(loss_function_class,onto)}"
        )
        print(f"Data Properties: {get_class_data_properties(onto,loss_function_class)}")
        print(f"Other Properties: {get_class_properties(onto, loss_function_class)}")

        # Create the "OptimizedEstimators" class as a disjoint sibling
        with onto:

            class OptimizedEstimators(Thing):
                equivalent_to = []
                is_a = [sup for sup in loss_function_class.is_a]  # Copy superclasses

        # Declare disjoint relationship
        loss_function_class.disjoint_with.append(OptimizedEstimators)

        # Save the modified ontology
        # onto.save(file="annett-o_additions.owl", format="rdfxml")
        print("OptimizedEstimators class created and saved successfully.")

    else:
        print("Loss Function class not found in the ontology.")


if __name__ == "__main__":
    onto = load_annetto_ontology()
    add_optimized_estimators(onto)
