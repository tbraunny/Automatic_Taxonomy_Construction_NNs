import os
import glob

from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
# from src.code_extraction.code_extractor import process_code_file
from src.instantiate_annetto.instantiate_annetto import instantiate_annetto

from utils.annetto_utils import load_annetto_ontology

def join_unique_dir(dir: str, dir2: str):
    """
    """

    path = os.path.join(dir, dir2)

    os.makedirs(path,exist_ok=True)

    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}_{i}"):
            i+=1
        path = f"{path}_{i}"
    
    os.makedirs(path)
    return path

def main(ann_name: str, ann_path:str,output_ontology_path: str) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :return: The path to the output ontology file.
    """
    
    
    
    # Extract text from PDF
    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    if ann_pdf_files:
        for pdf_file in ann_pdf_files:
            extract_filter_pdf_to_json(pdf_file, ann_path)
    
    # Extract code (give file path, glob is processed in the function)
    # process_code_file(ann_path)

    # Instantiate Annett-o
    if not os.path.exists(output_ontology_path):
        input_ontology = load_annetto_ontology(release_type="stable")
    else:
        input_ontology = load_annetto_ontology(owl_file_path=output_ontology_path)
        
    # output_ontology_path = load_annetto_ontology("test")
    instantiate_annetto(ann_name, ann_path, input_ontology, output_ontology_path)

    # return output_ontology_path

if __name__ == "__main__":
    main()