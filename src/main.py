import os
import glob
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from src.code_extraction.code_extractor import CodeExtractor
from src.instantiate_annetto.instantiate_annetto import instantiate_annetto
from src.instantiate_annetto.initialize_annetto import initialize_annetto
from utils.model_db_utils import DBUtils
from utils.owl_utils import delete_ann_configuration, save_ontology
from utils.constants import Constants as C
from utils.annetto_utils import load_annetto_ontology
import warnings
from typing import List

from utils.logger_util import get_logger

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
        raise ValueError("ANN name is required.")
    if not os.path.isdir(user_dir):
        raise ValueError("User_dir must be a directory.")
    
    owl_files = glob.glob(f"{user_dir}/*.owl") # should only be one owl file in a user folder
    if len(owl_files) > 1:
        warnings.warn(f"Multiple owl files found in {user_dir}. Using the first one.")
    owl_file = owl_files[0] if owl_files else None     
    if not owl_file:
        raise ValueError("No owl file found in the user_dir path.")
    
    ontology = load_annetto_ontology(return_onto_from_path=owl_file)
    with ontology:
        delete_ann_configuration(ontology, ann_name)
        save_ontology(ontology, owl_file)

def main(ann_name: str, ann_path: str, output_ontology_filepath: str = "", use_user_owl: bool = False) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :param use_user_owl: Whether to use the user-appended ontology or the pre-made stable ontology.
    """
    if not isinstance(ann_name, str):
        raise ValueError("ANN name must be a string.")
    if not os.path.isdir(ann_path):
        raise ValueError("ANN path must be a directory.")

    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    print(f"PDF files found in {ann_path}: {ann_pdf_files}")
    py_files = glob.glob(f"{ann_path}/**/*.py" , recursive=True)
    # onnx_files = glob.glob(f"{ann_path}/**/*.onnx" , recursive=True)
    # pb_files = glob.glob(f"{ann_path}/**/*.pb" , recursive=True)
    if not ann_pdf_files:
        print(f"No PDF files found in {ann_path}.")
        raise FileNotFoundError(f"No PDF files found in {ann_path}.")
    if not py_files: #and not onnx_files and not pb_files:
        raise FileNotFoundError(f"No code files found in {ann_path}.")
    # TODO: Use llm query to verify if the pdf is about a NN architecture
    # TODO: verify if code has any classes that inherit from nn.Module

    # Extract text from PDF, if any
    if ann_pdf_files:
        ann_doc_json = glob.glob(f"{ann_path}/*doc*.json")
        if not ann_doc_json:
                for pdf_file in ann_pdf_files:
                    extract_filter_pdf_to_json(pdf_file, ann_path)

    # Extract code (give file path, glob is processed in the function), if any
    pytorch_module_names: List[str] = []
    if py_files: # or onnx_files or pb_files:
        process_code = CodeExtractor()
        # ann_torch_json = glob.glob(f"{ann_path}/*torch*.json")
        # if not ann_torch_json:
        process_code.process_code_file(ann_path)
        pytorch_module_names = process_code.pytorch_module_names # for richie

    # # insert model into db
    # db_runner = DBUtils()
    # model_id: int = db_runner.insert_model_components(ann_path) # returns id of inserted model
    # paper_id: int = db_runner.insert_papers(ann_path)
    # translation_id: int = db_runner.model_to_paper(model_id, paper_id)

    output_ontology_filepath = C.ONTOLOGY.USER_OWL_FILENAME
    if not use_user_owl:
        input_ontology = load_annetto_ontology(return_onto_from_release="stable")
    else:
        input_ontology = load_annetto_ontology(
            return_onto_from_path=output_ontology_filepath
        )

    # Initialize Annett-o with new classes and properties
    print("Initializing Annett-o ontology...")
    initialize_annetto(input_ontology, logger)
    # Instantiate Annett-o
    instantiate_annetto(ann_name, ann_path, input_ontology, output_ontology_filepath, pytorch_module_names)

if __name__ == "__main__":
    # Example usage
    ann_name = "alexnet"
    user_path = "data/owl_testing"
    user_ann_path = os.path.join(user_path, ann_name)
    os.makedirs(user_ann_path, exist_ok=True)
    main(ann_name, user_ann_path, use_user_owl=False)
