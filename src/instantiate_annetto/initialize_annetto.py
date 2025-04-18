from owlready2 import Ontology
from utils.owl_utils import create_class_object_property, create_subclass

def add_has_weight_initialization(ontology: Ontology, logger=None) -> None:
    try:
        create_class_object_property(
            ontology, "hasWeightInitialization",
            ontology.TrainingSingle, ontology.WeightInitialization
        )
        if logger:
            logger.info("Object property 'hasWeightInitialization' added successfully.")
    except Exception as e:
        pass

def add_new_task_characterizations(ontology: Ontology, logger) -> None:
    new_classes = {
        "Self-Supervised Classification": ontology.TaskCharacterization,
        "Unsupervised Classification": ontology.TaskCharacterization,
    }
    for name, parent in new_classes.items():
        try:
            create_subclass(ontology, name, parent)
        except Exception as e:
            pass

def add_activation_function_layer_subclass(ontology: Ontology, logger=None) -> None:
    try:
        create_subclass(
            ontology, "ActivationFunctionLayer",
            ontology.Layer
        )
        if logger:
            logger.info("Subclass of Layer called 'ActivationFunctionLayer' added successfully.")
    except Exception as e:
        pass

def initialize_annetto(ontology: Ontology, logger=None) -> None:
    """Initialize annett-o ontology with new classes and properties."""
    try:
        add_has_weight_initialization(ontology, logger)
        add_new_task_characterizations(ontology, logger)
        add_activation_function_layer_subclass(ontology, logger)

        if logger:
            logger.info("Ontology initialized successfully.")
    except Exception as e:
        pass