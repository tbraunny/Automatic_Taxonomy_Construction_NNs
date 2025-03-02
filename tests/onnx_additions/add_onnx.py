# 1. connect to db
# 2. extract layers using tables.py from the db
# 3. instantiate annetto instances for the layers found in an ontology

"""
NOTE: 
- for the db connection to work, must forward port 5433 to virtual server postgres port (5432)
- put following in bash terminal:
    ssh -L 5433:172.20.199.232:5432 netid@nxlogin.engr.unr.edu
"""

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from utils.owl_utils import (
    create_cls_instance ,
    assign_object_property_relationship ,
    create_subclass ,
    get_subclasses ,
    print_instantiated_classes_and_properties ,
    get_object_properties_with_domain_and_range ,
)
from tables import Model , Layer , Parameter , Base
from utils.constants import Constants as C
from owlready2 import Ontology , ThingClass , Thing , ObjectProperty , get_ontology
import logging


class OnnxInstantiation:
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
        print(Model())
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
        """
        with self.engine.connect() as conn:
            layer = conn.execute(text("SELECT layer_name , known_type , model_id , attributes FROM layer"))
            model = conn.execute(text("SELECT model_id , model_name FROM model"))
            self.layer_list = layer.fetchall()
            self.model_list = model.fetchall()

    def instantiate_layer(self):
        # print("layer name: " , self.layer_list[0][0])
        # create_cls_instance(self.onto.Layer , self.layer_list[0][0])
        #create_subclass(self.onto , self.layer_list[0][0] , ThingClass)

        for name in self.layer_list:
            layer_name , type , model_id , attributes = name
            if not hasattr(self.onto , type): # condition never catches
                print(f"Creating subclass named {type}")
                parent = create_subclass(self.onto , type , self.onto.Layer)
            else:
                print("TYPE HERE: " , type)
                parent = getattr(self.onto.Layer , type)

            parent_name = f"{type}_instance"
            parent_instance = create_cls_instance(self.onto.Layer , parent_name)
            instance = create_cls_instance(parent , layer_name)

            object_property = get_object_properties_with_domain_and_range(self.onto , self.onto.ANNConfiguration , self.onto.Network)
            assign_object_property_relationship(parent_instance , instance , object_property)

            #logging.info(f"Instantiated layer {layer_name} as {type} with attributes ({attributes})")
        
            # parent = create_subclass(self.onto.Layer , name[1] , ThingClass)
            # instance = create_cls_instance(self.onto.Layer , name[0])
            # assign_object_property_relationship(parent , instance , name[1])

         # from parameter...
        # 1: name
        # 3: shape
        # 4: weights & biases

        # from layer...
        # 1: name
        # 2: layer type
        # 3: model_id
        # 4: attribs (parameters)
        ######### ONTO TESTING


def instantiate_onnx_annetto():
    onto_path="tests/onnx_additions/annetto-o-test.owl"
    inst = OnnxInstantiation()
    inst.init_engine()
    inst.init_onto(onto_path)
    inst.fetch_layers()
    inst.instantiate_layer()

    

if __name__ == '__main__':
    instantiate_onnx_annetto()