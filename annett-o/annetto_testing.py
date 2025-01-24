from owlready2 import *
from typing import List, Dict
from utils.utilsowl_parse import *

# # Load ontology

onto = get_ontology("./ontology/annett-o-0.1.owl").load()

class ConvolutionalNeuralNetwork(onto.ANNConfiguration):
    pass

# print(onto.ConvolutionalNeuralNetwork.is_a)

AlexNet = ConvolutionalNeuralNetwork("Alex_Net")

# print(get_class_properties_by_domain(onto, onto.ANNConfiguration))
# # Network = onto.Network

# AlexNet_Network = onto.Network("AlexNet_Network")

# print(get_class_restrictions(onto.ANNConfiguration)[0].type)
