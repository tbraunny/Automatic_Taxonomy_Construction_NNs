
import os
import glob
import warnings
from typing import List

import utils.suppress_init # used to suppress some default lib loggings
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from src.code_extraction.code_extractor import CodeExtractor
from src.ontology_population.populate_annetto import instantiate_annetto
from src.ontology_population.initialize_annetto import initialize_annetto
from src.taxonomy.create_taxonomy import TaxonomyCreator, TaxonomyNode, create_tabular_view_from_faceted_taxonomy, serialize
from src.taxonomy.criteria import *
from utils.model_db_utils import DBUtils
from utils.owl_utils import delete_ann_configuration, save_ontology
from utils.constants import Constants as C
from utils.annetto_utils import load_annetto_ontology
import warnings
from typing import List
import json
from src.taxonomy.llm_generate_criteria import llm_create_taxonomy

from utils.logger_util import get_logger

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import logging
logging.getLogger("faiss").setLevel(logging.ERROR)


logger = get_logger("main", max_logs=3)

def remove_ann_config_from_user_owl(ann_name: str, user_dir: str) -> None:
    """
    Removes the ANN configuration from the Annett-o ontology.
    Written specifically for user-appended ontologies.
    Saves the updated owl file to it's original location.

    :param ann_name: The name of the ANN.
    :param user_dir: The path to the directory containing the user files.
    """
    if not ann_name or not isinstance(ann_name, str):
        logger.error("ANN name is required.")
        raise ValueError("ANN name is required.")
    if not os.path.isdir(user_dir):
        logger.error("User_dir must be a directory.")
        raise ValueError("User_dir must be a directory.")
    
    owl_files = glob.glob(f"{user_dir}/*.owl") # should only be one owl file in a user folder
    if len(owl_files) > 1:
        logger.warning(f"Multiple owl files found in {user_dir}. Using the first one.")
        warnings.warn(f"Multiple owl files found in {user_dir}. Using the first one.")
    owl_file = owl_files[0] if owl_files else None     
    if not owl_file:
        logger.error("No owl file found in the user_dir path.")
        raise ValueError("No owl file found in the user_dir path.")
    ontology = load_annetto_ontology(return_onto_from_path=owl_file)
    with ontology:
        delete_ann_configuration(ontology, ann_name)
        save_ontology(ontology, owl_file)
        logger.info(f"Removed ANN configuration for {ann_name} from user owl file.")

def main(ann_name: str, ann_path: str, output_ontology_filepath: str = "", use_user_owl: bool = True) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :param output_ontology_filepath: The path to save the output ontology file, if you pass this, 
                                     it will overwrite the user onto file (lukas pls dont pass anything for this param, use the user_owl bool)
    :param use_user_owl: Whether to use the user-appended ontology or the pre-made stable ontology.
    """

    if not isinstance(ann_name, str):
        logger.error("ANN name must be a string.")
        raise ValueError("ANN name must be a string.")
    if not os.path.isdir(ann_path):
        logger.error("ANN path must be a directory.")
        raise ValueError("ANN path must be a directory.")

    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    py_files = glob.glob(f"{ann_path}/**/*.py" , recursive=True)
    # onnx_files = glob.glob(f"{ann_path}/**/*.onnx" , recursive=True)
    pb_files = glob.glob(f"{ann_path}/**/*.pb" , recursive=True)
    if not ann_pdf_files:
        logger.error(f"No PDF files found in {ann_path}.")
        raise FileNotFoundError(f"No PDF files found in {ann_path}.")
    if not py_files and not pb_files: #and not onnx_files:
        logger.error(f"No code files found in {ann_path}.")
        raise FileNotFoundError(f"No code files found in {ann_path}.")
    # TODO: Use llm query to verify if the pdf is about a NN architecture
    # TODO: verify if code has any classes that inherit from nn.Module

    # Extract text from PDF, if any
    if ann_pdf_files:
        ann_doc_json = glob.glob(f"{ann_path}/*doc*.json")
        if not ann_doc_json:
                for pdf_file in ann_pdf_files:
                    extract_filter_pdf_to_json(pdf_file, ann_path)
                    logger.info(f"Extracted text from {pdf_file} to JSON.")
    
    pytorch_module_names = []

    # # Extract code (give file path, glob is processed in the function), if any
    # pytorch_module_names: List[str] = []
    # if py_files or pb_files: # or onnx_files:
    #     process_code = CodeExtractor()
    #     # ann_torch_json = glob.glob(f"{ann_path}/*torch*.json")
    #     # if not ann_torch_json:
    #     process_code.process_code_file(ann_path)
    #     pytorch_module_names = process_code.pytorch_module_names # for richie
    #     logger.info(f"Extracted code from {py_files} to JSON.")

    #     # has_nn_module = process_code.pytorch_present
    #     # if not has_nn_module:
    #     #     logger.error("No nn.Module Pytorch classes found in the code files.")
    #     #     raise ValueError("No nn.Module Pytorch classes found in the code files.")

    # # # insert model into db
    # db_runner = DBUtils()
    # model_id: int = db_runner.insert_model_components(ann_path) # returns id of inserted model
    # paper_id: int = db_runner.insert_papers(ann_path)
    # translation_id: int = db_runner.model_to_paper(model_id, paper_id)

    if output_ontology_filepath:
        input_ontology = load_annetto_ontology(
            return_onto_from_path=output_ontology_filepath
        )
    elif not use_user_owl:
        input_ontology = load_annetto_ontology(return_onto_from_release="stable")
    else:
        input_ontology = load_annetto_ontology(
            return_onto_from_path=C.ONTOLOGY.USER_OWL_FILENAME
        )

    # Initialize Annett-o with new classes and properties
    initialize_annetto(input_ontology, logger)
    logger.info("Initialized Annett-o ontology.")

    # Instantiate Annett-o
    ontology_fp = instantiate_annetto(ann_name, ann_path, input_ontology, output_ontology_filepath, pytorch_module_names)
    logger.info("Instantiated Annett-o ontology.")

    # Define split criteria via llm
    ontology = load_annetto_ontology(return_onto_from_path=ontology_fp)
    thecriteria = llm_create_taxonomy('What would you say is the taxonomy that represents all neural network?', ontology)
    taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    format='json'
    topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(format=format,faceted=True)
    
    # Create faceted taxonomy as df
    df = create_tabular_view_from_faceted_taxonomy(taxonomy_str=json.dumps(serialize(facetedTaxonomy)), format=format)
    df.to_csv("./data/taxonomy/faceted/generic/generic_taxonomy.csv")

    
    

if __name__ == "__main__":
    # Example usage
    ann_name = "more_papers"
    user_path = "data/owl_testing"
    user_ann_path = os.path.join(user_path, ann_name)
    output_ontology_filepath = os.path.join(user_path, "user_owl.owl")
    os.makedirs(user_ann_path, exist_ok=True)
    main(ann_name, user_ann_path, output_ontology_filepath=output_ontology_filepath, use_user_owl=False)
