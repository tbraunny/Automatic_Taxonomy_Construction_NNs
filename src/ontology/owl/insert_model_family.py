""" 
Insert a model family class into annett-o owl file as a subclass of ANNConfiguration

Will later be used for dynamic ANNConfig subclass definitions
"""

from owlready2 import *
from typing import List, Dict
from utils.owl import create_subclass
from utils.constants import Constants as C

# Load ontology
onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

base_class = onto.ANNConfiguration

class_names = ["CNN"]

for class_name in class_names:
    #create_subclass adds to ontology namespace automatically
    new_class = create_subclass(ontology=onto, class_name=class_name, base_class=base_class)

onto.save(file='./data/owl/annett-o-0.2.owl') # Updated with ANNConfiguration Model Family Subclasses
