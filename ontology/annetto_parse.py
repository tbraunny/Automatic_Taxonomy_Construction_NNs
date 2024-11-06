from owlready2 import *
from typing import List, Dict

# # Load ontology
onto = get_ontology("./annett-o-0.1.owl").load()

# Access all instances of ANNConfiguration directly as Python objects
ann_config_instances = onto.ANNConfiguration.instances()

for ann_config in ann_config_instances:
    networks = ann_config.hasNetwork  # access related networks directly as attributes
    print(f"Networks for {ann_config.name}: {[net.name for net in networks]}")

