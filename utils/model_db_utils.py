import sqlalchemy as db
import logging
import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from rapidfuzz import process , fuzz

class DBUtils:
    """
    Utils for accessing, fetching & inserting data into remote Postgres server

    Must be run from WPEB machine to access server automatically
    """
    def __init__(self):
        self.engine , self.session = self._init_engine()
        self.layer_list = []
        self.model_list = []
        self.onto = 0

    def _init_engine(self):
        """
        Initialize & verify connection to the database

        :param None
        :return engine: connection engine for db comms
        :return session: session for connection to db
        """

        engine = db.create_engine('postgresql://postgres:postgres@100.80.229.100:5432/graphdb')
        Session = sessionmaker(bind=engine)
        session = Session()

        with engine.connect() as conn:
            try:
                total_networks = conn.execute(text("SELECT COUNT(graph) FROM model"))
                logging.info(f"DATABASE CONNECTED: Network count is {total_networks.fetchone()[0]}")
            except Exception as e:
                logging.exception(f"Failed to connect to database: {e}")

            return engine , session
        
    def __del__(self):
        self.session.close()
        
    def fetch_layers(self , network: str) -> list:
        """
        Fetch all relevant layer information from graphdb

        :param network: model network name
        :return layer_list: List of layers within the given network
        """
        try:
            query = text(f"SELECT l.layer_name, l.known_type, l.model_id, m.model_name FROM layer l JOIN model m ON l.model_id = m.model_id WHERE m.model_name = :network")
            layer_info = self.session.execute(query, {"network": network})
            self.layer_list: list = layer_info.fetchall()
        except Exception as e:
            logging.exception(f"Failed to fetch layers for {network} from database: {e}")
            self.layer_list = []

        return self.layer_list
    
    def fetch_all_models(self) -> list:
        """
        Fetch all models in database by id & name

        :param None
        :return model_list: List of models by id & name
        """
        try:
            query = text("SELECT model_name FROM model")
            models = self.session.execute(query)
            self.model_list: list[str] = [row[0] for row in models.fetchall()]
        except Exception as e:
            logging.exception(f"Failed to fetch models from database: {e}")
            self.model_list = []

        return self.model_list
    
    def insert_model(self , name: str , model_type: str , graph: json , avg_embedding) -> int:
        """
        Insert a model into the database

        :param name: Name of the model
        :param type: Task characterization of the model?
        :param graph: Symbolic graph of the model (json in data/{ann_name}/*.json)
        :param avg_embedding: Average weight embedding of the model
        :return model_id: Assigned model_id of the inserted model
        """
        model_id = -1
        try:
            if not avg_embedding:
                query = text("""INSERT INTO model (model_name , library , graph) 
                            VALUES (:name , :model_type , :graph) 
                            RETURNING model_id""")
                result = self.session.execute(query , {
                    "name": name,
                    'model_type': model_type,
                    "graph": graph
                })
            else:
                query = text("""INSERT INTO model (model_name , library , average_weight_embedding , graph) 
                            VALUES (:name , :model_type , :graph , :avg_embedding) 
                            RETURNING model_id""")
                result = self.session.execute(query , {
                    "name": name,
                    'model_type': model_type,
                    "avg_embedding": avg_embedding,
                    "graph": graph
                })
            model_id = result.scalar()
            self.session.commit()
        except Exception as e:
            logging.exception(f"Failed to insert model {name} into the database: {e}")

        return model_id
    
    def insert_layer(self , model_id: int , name: str , layer_type: str , attributes: dict) -> int:
        """
        Inserts layer into the database for a specific model

        :param model_id: Model ID
        :param name: Name of the layer
        :param type: Operation type of the layer
        :param attributes: Relevant attributes
        :return layer_id: Assigned ID of the inserted layer
        """
        layer_id = -1
        try:
            query = text("""INSERT INTO layer (layer_name , known_type , model_id , attributes)
                         VALUES (:name , :layer_type , :model_id , :attributes)
                         RETURNING layer_id""")
            result = self.session.execute(query , {
                "name": name,
                "layer_type": layer_type,
                "model_id": model_id,
                "attributes": attributes
            })
            layer_id = result.scalar()
            self.session.commit()
        except Exception as e:
            logging.exception(f"Failed to insert layer {name} in model {model_id} into database: {e}")
            
        return layer_id
    
    def insert_parameter(self , layer_id: int , name: str , shape: tuple , weight_embedding) -> None:
        """
        Insert parameter into the database

        :param layer_id: Layer ID
        :param name: Name of the parameter
        :param shape: Shape of the parameter
        :param weight_embedding: Weight embedding vector of the parameter
        :return None
        """
        try:
            query = text("""INSERT INTO parameter (name , layer_id , shape , weight_embedding)
                         VALUES (:name , :layer_id , :shape , :weight_embedding)""")
            self.session.execute(query , {
                "name": name,
                "layer_id": layer_id,
                "shape": shape,
                "weight_embedding": weight_embedding
            })
            self.session.commit()
        except Exception as e:
            logging.exception(f"Failed to insert parameter {name} into layer {layer_id} in the database: {e}")

    def insert_papers(self , paper_name: str , contents: str) -> int:
        """
        Insert paper for model into the database

        :param paper_name: Name of the paper (actual title)
        :param contents: The contents of the paper (bytea)
        :return paper_id: Insert paper's ID
        """
        paper_id = -1
        try:
            query = text("""INSERT INTO paper (paper_name , contents)
                         VALUES (:paper_name , :contents)
                         RETURNING paper_id""")
            result = self.session.execute(query , {
                "paper_name": paper_name,
                "contents": contents
            })
            paper_id = result.scalar()
            self.session.commit()
        except Exception as e:
            logging.exception(f"Failed to insert paper {paper_name} into database: {e}")

        return paper_id
    
    def model_to_paper(self , model_id: int , paper_id: int) -> int:
        """
        Insert translation between model_id & paper_id

        :param model_id: Model ID for the corresponding paper
        :param paper_id: Paper ID for the corresponding model
        :return paper_model_id: Translation ID
        """
        paper_model_id = -1
        try:
            query = text("""INSERT INTO paper_model (paper_id , model_id)
                         VALUES (:paper_id , :model_id)
                         RETURNING paper_model_id""")
            result = self.session.execute(query , {
                "paper_id": paper_id,
                "model_id": model_id
            })
            paper_model_id = result.scalar()
        except Exception as e:
            logging.exception(f"Failed to translate model {model_id} to paper {paper_id}: {e}")

        return paper_model_id

    def find_model(self , name: str=None , model_id: int=None) -> str:
        """
        Find a model in the dataabase given its ID (if no ID, fuzzy match by name)

        :param name: Name of the model
        :param model_id: ID for the model
        :return model name
        """
        if model_id:
            query = text("""SELECT name FROM model WHERE model_id = :model_id""")
            result = self.session.execute(query , {
                "model_id": model_id
            })
            model = result.fetchone()
        elif name:
            # fuzzy match for the name
            pass
        else:
            raise "name or model_id params requried"

        return model

def insert_model(ann_name: str , model_json: json) -> None:
    """
    Insert a model into the database given its symbolic graph (json)

    :param ann_name: Name of model/network
    :param model_json: JSON file of model to be inserted
    :return None
    """
    insert = DBUtils()
    network_data: dict = json.loads(model_json)
    nodes = network_data.get('graph' , {}).get('node' , {})

    model_type = None

    model_id: int = insert.insert_model(ann_name , model_type , model_json)

    for layer in nodes:
        layer_name = layer.get('name')
        layer_type = layer.get('op_type')
        layer_attr = layer.get('attributes')

        layer_id = insert.insert_layer(model_id , layer_name , layer_type , layer_attr)
        shape = layer.get('kernel')

        if layer_id and shape:
            insert.insert_parameter(layer_id , 'kerne;' , shape)

    
if __name__ == '__main__': # example usage
    ann_name = "alexnet"
    runner = DBUtils()