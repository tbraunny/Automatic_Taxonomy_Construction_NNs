from owlready2 import Ontology
from utils.owl_utils import create_class_object_property, create_subclass, create_generic_data_property

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
            if logger:
                logger.info(f"Subclass of TaskCharacterization called '{name}' added successfully.")
        except Exception as e:
            logger.debug(f"Failed to add subclass '{name}': {e}",exc_info=True) if logger else None
            pass

def add_activation_function_layer_subclass(ontology: Ontology, logger=None) -> None:
    try:
        create_subclass(
            ontology, "ActivationFunctionLayer",
            ontology.Layer
        )
        if logger:
            logger.info("Subclass of Layer called 'ActivationFunctionLayer' added successfully.",exc_info=True)
    except Exception as e:
        pass
def add_source_data_property(ontology: Ontology, logger=None) -> None:
    try:
        prop = create_generic_data_property(
            ontology, "sourceData",
            str
        )
        print(f"Property created: {prop}")
        if logger:
            logger.info("Data property 'sourceData' added successfully.")
    except Exception as e:
        logger.debug(f"Failed to add data property 'sourceData': {e}",exc_info=True) if logger else None
        pass
def add_definition_data_property(ontology: Ontology, logger=None) -> None:
    try:
        create_generic_data_property(
            ontology, "definition",
            str
        )
        if logger:
            logger.info("Data property 'definition' added successfully.")
    except Exception as e:
        logger.debug(f"Failed to add data property 'definition': {e}",exc_info=True) if logger else None
        pass
def remove_annetto_default_anns(ontology: Ontology, logger=None) -> None:
    try:
        from utils.owl_utils import delete_ann_configuration
        delete_ann_configuration(ontology, "GAN")
        delete_ann_configuration(ontology, "AAE")
        delete_ann_configuration(ontology, "simple_classification")
        if logger:
            logger.info("Default ANN configurations removed successfully.")
    except Exception as e:
        logger.debug(f"Failed to remove default ANN configurations: {e}",exc_info=True) if logger else None
        pass

def initialize_annetto(ontology: Ontology, logger=None) -> None:
    """Initialize annett-o ontology with new classes and properties."""
    try:
        add_has_weight_initialization(ontology, logger)
        add_new_task_characterizations(ontology, logger)
        add_activation_function_layer_subclass(ontology, logger)
        remove_annetto_default_anns(ontology, logger)
        add_source_data_property(ontology, logger)
        add_definition_data_property(ontology, logger)
        # Add any other initialization steps here

        if logger:
            logger.info("Ontology initialized successfully.")
            print("Ontology initialized successfully.")
    except Exception as e:
        logger.debug(f"Failed to make initial ontology updates: {e}") if logger else None
        pass