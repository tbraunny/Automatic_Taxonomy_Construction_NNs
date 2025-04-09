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

    def fetch_layers(self , network):
        """
        Fetch all relevant layer information from graph db
        Returns lists of all layers w model names
        """
        with self.engine.connect() as conn:
            layer_info = conn.execute(text(f"SELECT l.layer_name, l.known_type, l.model_id, m.model_name FROM layer l JOIN model m ON l.model_id = m.model_id WHERE m.model_name = :network") , {"network": network})
            self.layer_list:List = layer_info.fetchall()

        return self.layer_list

    def fetch_models(self):
        """
        Fetch all models in the database by id & name
        """
        with self.engine.connect() as conn:
            models = conn.execute(text("SELECT model_name FROM model"))
            #self.model_list:Union[int , List[int] , str , List[str]] = models.fetchall()
            self.model_list: List[str] = [row[0] for row in models.fetchall()]  # Extract the first column


        return self.model_list

def instantiate_onnx_annetto():
    # testing
    onto_path="tests/onnx_additions/annetto-o-test.owl"
    inst = OnnxAddition()
    inst.init_engine()
    inst.fetch_layers(network="alexnet")
    

if __name__ == '__main__':
    instantiate_onnx_annetto()