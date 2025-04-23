from owlready2 import *
from utils.annetto_utils import load_annetto_ontology
from utils.owl_utils import get_class_instances
from utils.constants import Constants as C

def list_of_class_instances():
    ontology_filepath = C.ONTOLOGY.USER_OWL_FILENAME
    ontology = load_annetto_ontology(return_onto_from_path=ontology_filepath)
    instances = get_class_instances(ontology.ANNConfiguration)

    return instances
    
    
print(list_of_class_instances())