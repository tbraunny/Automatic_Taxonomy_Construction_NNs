# 1. connect to db
# 2. extract layers using tables.py from the db
# 3. instantiate annetto instances for the layers found in an ontology

"""
NOTE: 
- for the db connection to work, must forward port 5433 to virtual server postgres port (5432)
- put following in bash terminal:
    ssh -L 5433:172.20.199.232:5432 netid@nxlogin.engr.unr.edu
"""
from typing import List, Union, Dict

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from utils.owl_utils import (
    create_cls_instance ,
    assign_object_property_relationship ,
    create_subclass ,
    print_instantiated_classes_and_properties ,
    get_object_properties_with_domain_and_range ,
    get_all_subclasses
)
#from tables import Model , Layer , Parameter , Base
from utils.constants import Constants as C
from owlready2 import Ontology , ThingClass , Thing , ObjectProperty , get_ontology
import logging
import utils.annetto_utils as ann


class OnnxAddition:
    """
    Fetch ONNX layers from graph database to instantiate ANNETT-O
    layers for all models present
    """

    def __init__(self):
        self.engine = 0
        self.session = 0
        self.layer_list = []
        self.model_list = []
        self.onto = 0

    def init_engine(self):
        """
        Initialize connection to the database
        """
        #print(Model())
        self.engine = db.create_engine('postgresql://postgres:postgres@localhost:5433/graphdb')
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        with self.engine.connect() as conn:
            try:
                total_networks = conn.execute(text("SELECT COUNT(graph) FROM model"))
                logging.info(f"DATABASE CONNECTED: Network count is {total_networks.fetchone()[0]}")
            except Exception as e:
                logging.exception(f"Failed to connect to database: {e}")

    def init_onto(self , new_path="tests/onnx_additions/annetto-o-test.owl"):
        """
        Initialize ontology
        """
        onto_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
        self.onto = get_ontology(onto_path).load()

        with self.onto: 
            try:
                self.onto.save(file=new_path , format="rdfxml")
                logging.info(f"Ontology saved to {new_path}")
            except Exception as e:
                logging.excpetion(f"Error saving ontology: {e}")

    def fetch_layers(self):
        """
        Fetch all relevant layer information from graph db
        Returns lists of all layers & all models
        """
        with self.engine.connect() as conn:
            layer = conn.execute(text("SELECT layer_name , known_type , model_id , attributes FROM layer"))
            model = conn.execute(text("SELECT model_id , model_name FROM model"))
            self.layer_list:List = layer.fetchall()
            self.model_list:Union[str , List[str] , int , List[int]] = model.fetchall()

        return self.layer_list , self.model_list


def instantiate_onnx_annetto():
    onto_path="tests/onnx_additions/annetto-o-test.owl"
    inst = OnnxAddition()
    inst.init_engine()
    inst.init_onto(onto_path)
    inst.fetch_layers()
    

if __name__ == '__main__':
    instantiate_onnx_annetto()