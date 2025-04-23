import sqlalchemy as db
import json
import glob
import os
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from rapidfuzz import process
from utils.logger_util import get_logger
from pathlib import Path

load_dotenv()
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

class DBUtils:
    """
    Utils for accessing, fetching & inserting data into remote Postgres server

    Must be run from WPEB machine to access server automatically (Tailscale network)
    """
    def __init__(self):
        self.logger = get_logger("model-db-utils")
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

        engine = db.create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        with engine.connect() as conn:
            try:
                total_networks = conn.execute(text("SELECT COUNT(graph) FROM model"))
                self.logger.info(f"DATABASE CONNECTED: Network count is {total_networks.fetchone()[0]}")
            except Exception as e:
                self.logger.exception(f"Failed to connect to database: {e}")

            return engine , session
        
    def __del__(self):
        self.session.close()
        
    def _fetch_layers(self , network: str) -> list:
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
            self.logger.exception(f"Failed to fetch layers for {network} from database: {e}")
            self.layer_list = []

        return self.layer_list
    
    def _fetch_all_models(self) -> list:
        """
        Fetch all models in database by name

        :param None
        :return model_list: List of models by name
        """
        try:
            query = text("SELECT model_name FROM model")
            models = self.session.execute(query)
            self.model_list: list[str] = [row[0] for row in models.fetchall()]
        except Exception as e:
            self.logger.exception(f"Failed to fetch models from database: {e}")
            self.model_list = []

        return self.model_list
    
    def _insert_model(self , name: str , graph: json , avg_embedding=None , model_type: str=None) -> int:
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
                    "graph": json.dumps(graph) if graph else None
                })
            else:
                query = text("""INSERT INTO model (model_name , library , average_weight_embedding , graph) 
                            VALUES (:name , :model_type , :graph , :avg_embedding) 
                            RETURNING model_id""")
                result = self.session.execute(query , {
                    "name": name,
                    'model_type': model_type,
                    "avg_embedding": avg_embedding,
                    "graph": json.dumps(graph) if graph else None
                })
            model_id = result.scalar()

            self.session.commit()
        except Exception as e:
            self.logger.exception(f"Failed to insert model {name} into the database: {e}")

        return model_id

    def insert_model_type(self , model_id: int , model_type: str) -> None:
        """
        Insert model type into the db if it was not previously known
        """
        try:
            query = text("""INERT INTO model (type) 
                         VALUES (:model_type) 
                         WHERE model_id = :model_id""")
            self.session.execute(query , {
                "model_type": model_type,
                "model_id": model_id
            })
            self.session.commit()
        except Exception as e:
            self.logger.exception(f"Failed to insert model {model_id} type {model_type} into the database: {e}")

    def _insert_layer(self , model_id: int , name: str , layer_type: str , attributes: dict) -> int:
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
                "attributes": json.dumps(attributes) if attributes else None
            })
            layer_id = result.scalar()
            self.session.commit()
        except Exception as e:
            self.logger.exception(f"Failed to insert layer {name} in model {model_id} into database: {e}")
        
        return layer_id
    
    def _insert_parameter(self , layer_id: int , name: str , shape: tuple , weight_embedding=None) -> None:
        """
        Insert parameter into the database

        :param layer_id: Layer ID
        :param name: Name of the parameter
        :param shape: Shape of the parameter
        :param weight_embedding: (Optional) Weight embedding vector of the parameter
        :return None
        """
        try:
            query = text("""INSERT INTO parameter (parameter_name , layer_id , shape , weight_embedding)
                         VALUES (:name , :layer_id , :shape , :weight_embedding)""")
            self.session.execute(query , {
                "name": name,
                "layer_id": layer_id,
                "shape": shape,
                "weight_embedding": weight_embedding
            })
            self.session.commit()
        except Exception as e:
            self.logger.exception(f"Failed to insert parameter {name} into layer {layer_id} in the database: {e}")

    def find_model_id(self , name: str) -> int:
        """
        Find a model in the database its name via fuzzy match

        :param name: Name of the model
        :return Model ID
        """
        model_list: list = self._fetch_all_models()
        model_list = [model.lower() for model in model_list]
        model_name , score , _ = process.extractOne(name.lower() , model_list) # high score due to similar names in db

        if not model_name or score < 90:
            self.logger.warning(f"Could not find an existing model in the database matching {name}")

        query = text("""SELECT model_id 
                     FROM model 
                     WHERE model_name = :model_name""")
        result = self.session.execute(query , {
            "model_name": model_name
        })
        model_id: int = result.scalar()

        return model_id

    def insert_papers(self , ann_path: str) -> int:
        """
        Insert paper for model into the database

        :param paper_name: Name of the paper (actual title)
        :return paper_id: Insert paper's ID
        """
        paper_id = -1
        try:
            pdf_files: list = glob.glob(f"{ann_path}/**/*.pdf" , recursive=True)

            for pdf in pdf_files:
                contents: dict = {}
                with open(pdf , 'rb') as f:
                    contents = f.read()

                query = text("""INSERT INTO paper (paper_name , contents)
                            VALUES (:paper_name , :contents)
                            RETURNING paper_id""")
                result = self.session.execute(query , {
                    "paper_name": ann_path,
                    "contents": contents
                })
                paper_id = result.scalar()
                self.session.commit()
        except Exception as e:
            self.logger.exception(f"Failed to insert paper {ann_path} into database: {e}")

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
            self.session.commit()
        except Exception as e:
            self.logger.exception(f"Failed to translate model {model_id} to paper {paper_id}: {e}")

        return paper_model_id

    def _find_model_name(self , name: str=None , model_id: int=None) -> str:
        """
        Find a model in the dataabase given its ID (if no ID, fuzzy match by name)

        :param name: Name of the model
        :param model_id: ID for the model
        :return model name
        """
        if model_id:
            query = text("""SELECT name 
                         FROM model 
                         WHERE model_id = :model_id""")
            result = self.session.execute(query , {
                "model_id": model_id
            })
            model = result.fetchone()
        elif name: # fuzzy match for similar name
            name = name.lower()
            model_list: list = self._fetch_all_models()
            model , _ , _ = process.extractOne(name , model_list , score_cutoff=90)
        else:
            raise "Name and/or model_id params requried"

        return model

    def fetch_layers_for_model(self , name: str=None , model_id: int=None) -> dict:
        """
        Return dictionary of layers that comprise a given model

        :param name: Name of the model
        :param model_id: ID of the model (if known)
        :return Dictionary of layers
        """
        layers: dict = {
            "layer_type": None,
            "attributes": {}
        }

        if not model_id:
            model_id: int = self.find_model_id(name)
        
        query = text("""SELECT layer_name , known_type , attributes
                     FROM layer 
                     WHERE model_id = :model_id""")
        result = self.session.execute(query , {
            "model_id": model_id
        })
        for element in result.fetchall():
            layers[element[0]] = {
                "layer_type": element[1],
                "attributes": element[2]
            }

        return layers

    def insert_model_components(self , ann_path: str) -> int:
        """
        Insert a model into the database given its symbolic graph (json) & components

        :param ann_name: Name of model/network
        :return model_id: ID of the inserted model
        """
        json_files: list = [] # fetch relevant jsons from ann_path
        json_files.extend(glob.glob(f"{ann_path}/**/*torch*.json" , recursive=True))
        json_files.extend(glob.glob(f"{ann_path}/**/*pb*.json" , recursive=True))
        json_files.extend(glob.glob(f"{ann_path}/**/*onnx*.json" , recursive=True))

        if not json_files:
            raise FileNotFoundError

        for file in json_files:
            network_data: dict = {}

            with open(file , 'r') as f:
                network_data: dict = json.load(f)
            
            model_name = os.path.basename(ann_path)
            model_id: int = self._insert_model(model_name , network_data)
            nodes = network_data.get("graph" , {}).get("node" , {})

            for layer in nodes:
                layer_name = layer.get('name')
                layer_type = layer.get('op_type')
                layer_attr = layer.get('attributes')

                layer_id = self._insert_layer(model_id , layer_name , layer_type , layer_attr)
                shape = layer.get('kernel')

                if layer_id and layer_attr:
                    for attr in layer_attr:
                        for shape in attr:
                            self._insert_parameter(layer_id , attr , shape)

        return model_id

    
if __name__ == '__main__':
    ann_name = "alexnet"
    ann_path = f"data/{ann_name}"
    runner = DBUtils()
    runner.fetch_layers_for_model(ann_name)
    # model_id = runner.insert_model_components(ann_path)
    # paper_id = runner.insert_papers(ann_path)
    # model_paper_id = runner.model_to_paper(model_id , paper_id)

    # print("MODEL: " , model_id)
    # print("PAPER: " , paper_id)
    # print("MODEL PAPER: " , model_paper_id)