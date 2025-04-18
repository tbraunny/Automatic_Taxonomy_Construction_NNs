from owlready2 import *
from utils.annetto_utils import load_annetto_ontology
from utils.owl_utils import get_class_instances

def list_of_class_instances():
    ontology = load_annetto_ontology(return_onto_from_release="stable")
    instances = get_class_instances(ontology.ANNConfiguration)
    # outowl =""
    
    # ontology = load_annetto_ontology(owl_file_path=outowl)
    
    return instances
    