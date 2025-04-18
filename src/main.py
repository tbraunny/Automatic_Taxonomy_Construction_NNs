import os
import glob

from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from src.code_extraction.code_extractor import CodeExtractor
from src.instantiate_annetto.instantiate_annetto import instantiate_annetto
from utils.model_db_utils import DBUtils
from utils.owl_utils import delete_ann_configuration, save_ontology
from utils.constants import Constants as C

from utils.annetto_utils import load_annetto_ontology

def remove_ann_config_from_user_owl(hashed_ann_name: str, ann_path: str) -> None:
    """
    Removes the ANN configuration from the Annett-o ontology.
    Written specifically for user-appended ontologies.
    Saves the updated owl file to it's original location.

    :param hashed_ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    """
    if not hashed_ann_name:
        raise ValueError("ANN name is required.")
    if not "_" in hashed_ann_name:
        raise ValueError("ANN name must contain a hash.")
    if not os.path.isdir(ann_path):
        raise ValueError("ANN path must be a directory.")
    
    owl_file = glob.glob(f"{ann_path}/*.owl")[0] # should only be one owl file in a user folder
    if not owl_file:
        raise ValueError("No owl file found in the ANN path.")
    
    ontology = load_annetto_ontology(return_onto_from_path=owl_file)
    with ontology:
        delete_ann_configuration(hashed_ann_name)
        save_ontology(owl_file)

def main(ann_name: str, ann_path: str, use_user_owl: bool = False) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :param use_user_owl: Whether to use the user-appended ontology or the pre-made stable ontology.
    :return: The hashed ANNConfig instance name for later use.
    """
    if not isinstance(ann_name, str):
        raise ValueError("ANN name must be a string.")
    if not os.path.isdir(ann_path):
        raise ValueError("ANN path must be a directory.")

    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    py_files = glob.glob(f"{ann_path}/**/*.py" , recursive=True)
    onnx_files = glob.glob(f"{ann_path}/**/*.onnx" , recursive=True)
    pb_files = glob.glob(f"{ann_path}/**/*.pb" , recursive=True)

    # Check if any pdfs were found
    if not ann_pdf_files:
        print(f"No PDF files found in {ann_path}.")
    # Check if any code files were found
    if not py_files and not onnx_files and not pb_files:
        raise FileNotFoundError(f"No files found in {ann_path}.")

    # Extract text from PDF, if any
    if ann_pdf_files:
        for pdf_file in ann_pdf_files:
            extract_filter_pdf_to_json(pdf_file, ann_path)

    # Extract code (give file path, glob is processed in the function), if any
    process_code = CodeExtractor()
    pytorch_module_names: list = []
    if py_files or onnx_files or pb_files:
        process_code.process_code_file(ann_path)
        pytorch_module_names = process_code.pytorch_module_names # for richie

    # insert model into db
    db_runner = DBUtils()
    model_id = db_runner.insert_model_components(ann_path) # returns id of inserted model

    output_ontology_filepath = os.path.join(ann_path, C.ONTOLOGY.USER_OWL_FILENAME) # User owl file always uses this name
    if not use_user_owl:
        input_ontology = load_annetto_ontology(return_onto_from_release="stable")
    else:
        input_ontology = load_annetto_ontology(
            return_onto_from_path=output_ontology_filepath
        )
    # Instantiate Annett-o
    hashed_ann_name = instantiate_annetto(ann_name, ann_path, input_ontology, output_ontology_filepath)
    
    # Returns the hashed ANN name
    if not hashed_ann_name:
        raise ValueError("Failed to instantiate Annett-o.")
    print(f"Successfully instantiated Annett-o with name: {hashed_ann_name}")
    return hashed_ann_name

if __name__ == "__main__":
    # Example usage
    ann_name = "alexnet"
    user_path = "data/owl_testing"
    user_ann_path = os.path.join(user_path, ann_name)
    assert glob.glob(f"{user_ann_path}/*.pdf") # Makes sure a pdf is in the dir
    main(ann_name, user_ann_path, use_user_owl=False)