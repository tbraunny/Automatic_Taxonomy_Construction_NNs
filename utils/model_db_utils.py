import sqlalchemy as db
import logging
import json
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from rapidfuzz import process , fuzz

class DBUtils:
    def __init__(self):
        self.engine , self.session = self._init_engine()
        self.layer_list = []
        self.model_list = []
        self.onto = 0
    
    def __del__(self):
        self.session.close()

    def _init_engine(self):
        """
        Initialize & verify connection to the database

        :param None
        :return engine: connection engine for db comms
        :return session: session for connection to db
        """

        engine = db.create_engine('postgresql://postgres:postgres@localhost:5433/graphdb')
        Session = sessionmaker(bind=self.engine)
        session = Session()

        with self.engine.connect() as conn:
            try:
                total_networks = conn.execute(text("SELECT COUNT(graph) FROM model"))
                logging.info(f"DATABASE CONNECTED: Network count is {total_networks.fetchone()[0]}")
            except Exception as e:
                logging.exception(f"Failed to connect to database: {e}")

            return engine , session
        
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
    
    def fetch_models(self) -> list:
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
    
    def insert_model(self , name: str , type: str , graph: json) -> int:
        """
        Insert a model into the database

        :param name: Name of the model
        :param type: Task characterization of the model?
        :param graph: Symbolic graph of the model (json in data/{ann_name}/*.json)
        :return model_id: Assigned model_id of the inserted model
        """
        model_id = -1
        try:
            query = text("")
        except Exception as e:
            logging.exception(f"Failed to insert model {name} into the database: {e}")

        return model_id

    def insert_papers(self , paper_name: str , contents: str) -> None:
        """
        Insert paper for model into the database

        :param paper_name: Name of the paper (actual title)
        :param contents: The contents of the paper (bytea)
        :return None
        """
        try:
            query = text("")
        except Exception as e:
            logging.exception(f"Failed to insert paper {paper_name} into database: {e}")

    
if __name__ == '__main__':
    ann_name = "alexnet"
    runner = DBUtils()
    runner._init_engine() # is this necessary?
    print(runner.fetch_layers(network=ann_name))
    print(runner.fetch_models())