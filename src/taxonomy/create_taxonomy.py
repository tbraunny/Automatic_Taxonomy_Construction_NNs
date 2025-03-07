from utils.owl_utils import *
from utils.annetto_utils import make_thing_classes_readable
from utils.constants import Constants as C
from pathlib import Path
import logging


# Set up logging @ STREAM level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TaxonomyCreator:
    def __init__(self, ontology: Ontology):
        self.ontology = ontology # Load ontology as Ontology object
        self.taxonomy = None
    
    def create_taxonomy(self):
        # Get all ANNConfiguration Objects
        logger.info(f"ANNConfiguration Class: {self.ontology.ANNConfiguration}, type: {type(self.ontology.ANNConfiguration)}")

        ann_configurations = get_class_instances(self.ontology.ANNConfiguration)
        logger.info(f"ANNConfigurations: {ann_configurations}, type: {type(ann_configurations)}")

        for ann_config in ann_configurations:
            logger.info(f"{" " * 3}ANNConfig: {ann_config}, type: {type(ann_config)}")
            # NOTE: ontology.hasNetwork is an ObjectProperty -> returns annett-o-0.1.hasNetwork of type: <class 'owlready2.prop.ObjectPropertyClass'>
            networks = get_instance_property_values(ann_config, self.ontology.hasNetwork.name)

            for network in networks:
                logger.info(f"{" " * 5}Network: {network}, type: {type(network)}")

                task_characterizations = get_instance_property_values(network, self.ontology.hasTaskType.name)
                logger.info(f"{" " * 5}Task Characterizations: {task_characterizations}, type: {type(task_characterizations)}")

                layers = get_instance_property_values(network, self.ontology.hasLayer.name)
                logger.info(f"{" " * 5}Layers: {layers}, type: {type(layers)}")

                for layer in layers:
                    # NOTE: Here we can access the class (ie for layer the subclass we care about) in two ways, we use .is_a[0] more typically
                    logger.info(f"{" " * 7}Layer: {layer}, type: {type(layer)}")
                    logger.info(f"{" " * 7}Layer: {layer}, type: {layer.is_a}")
                    layer_type = get_instance_property_values(layer, self.ontology.hasLayer.name)
                    logger.info(f"{" " * 7}Layer Type: {layer_type}, type: {type(layer_type)}")

                for task_characterization in task_characterizations:
                    logger.info(f"{" " * 7}Task Characterization: {task_characterization}, type: {type(task_characterization)}")
                    logger.info(f"{" " * 7}Task Characterization: {task_characterization}, type: {task_characterization.is_a}")

                    subclass = task_characterization.is_a[0]
                    logger.info(f"{" " * 7}Subclass: {subclass}, type: {type(subclass)}")


def main():

    logger.info("Loading ontology.")
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}" 
    # ontology_path = f"./data/owl/annett-o-test.owl"

    ontology = load_ontology(ontology_path=ontology_path)
    logger.info("Ontology loaded.")

    logger.info("Creating taxonomy from Annetto annotations.")
    taxonomy_creator = TaxonomyCreator(ontology)
    taxonomy_creator.create_taxonomy()
    logger.info("Finished creating taxonomy.")


if __name__ == "__main__":
    main()