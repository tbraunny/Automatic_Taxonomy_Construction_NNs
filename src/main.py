import os
import glob

from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from src.code_extraction.code_extractor import process_code_file
from src.instantiate_annetto.instantiate_annetto import instantiate_annetto
import utils.constants as C
from utils.annetto_utils import load_annetto_ontology

def create_ann_dir(ann_name: str):
    """
    Creates a directory for the ANN in the data/ directory.
    """
    base_dir = "data"

    path = os.path.join(base_dir, ann_name)

    os.makedirs(base_dir,exist_ok=True)

    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}_{i}"):
            i+=1
        path = f"{path}_{i}"
    
    os.makedirs(path)
    return path

def main(ann_name: str, ann_path:str) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :return: The path to the output ontology file.
    """
    
    # Make a directory for the ANN # TEMP
    # ann_path = create_ann_dir(ann_name)

    # TODO: This is where front-end will put user files

    # Extract text from PDF
    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    if ann_pdf_files:
        for pdf_file in ann_pdf_files:
            extract_filter_pdf_to_json(pdf_file, ann_path)


    # Instantiate Annett-o
    input_ontology_path = load_annetto_ontology("base")
    output_ontology_path = C.ONTOLOGY.TEST_ONTOLOGY_PATH
    instantiate_annetto(ann_name, ann_path, input_ontology_path, output_ontology_path)

    return output_ontology_path

if __name__ == "__main__":
    main()