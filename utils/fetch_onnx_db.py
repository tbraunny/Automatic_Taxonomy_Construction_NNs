# 1. connect to db
# 2. extract layers using tables.py from the db
# 3. instantiate annetto instances for the layers found in an ontology

"""
NOTE: 
- for the db connection to work, must forward port 5433 to virtual server postgres port (5432)
- put following in bash terminal:
    ssh -L 5433:172.20.199.232:5432 netid@nxlogin.engr.unr.edu
"""
from typing import List

import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from rapidfuzz import process , fuzz
import logging
from typing import Optional

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
    
    def _fuzzy_match_list(self , class_names: List[str], instance=None , threshold: int = 80) -> Optional[str]:
        """
        Perform fuzzy matching to find the best match for an instance in a list of strings.

        :param instance_name: The instance name.
        :param class_names: A list of string names to match with.
        :param threshold: The minimum score required for a match.
        :return: The best-matching string or None if no good match is found.
        """
        if not instance:
            print("WHAT THE FUCK")

        if not all(isinstance(name, str) for name in class_names):
            raise TypeError("Expected class_names to be a list of strings.")
        if not isinstance(threshold, int):
            raise TypeError("Expected threshold to be an integer.")

        match, score, _ = process.extractOne(self.ann_config_name, class_names, scorer=fuzz.ratio)

        return match if score >= threshold else None

    def check_onnx(self , model_name):
        try:
            onn = OnnxAddition()
            onn.init_engine()
            models_list = onn.fetch_models()
            best_model_match = self._fuzzy_match_list(models_list , model_name)
            return best_model_match
        except Exception as e:
            raise e

def fetch_db_info():
    # testing
    ann_name = "alexnet"
    inst = OnnxAddition()
    inst.init_engine()
    print(inst.fetch_layers(network=ann_name))
    print(inst.fetch_models())
    

if __name__ == '__main__':
    fetch_db_info()