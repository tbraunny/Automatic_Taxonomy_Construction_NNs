import os
import glob

from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json
from src.code_extraction.code_extractor import process_code_file
from src.instantiate_annetto.instantiate_annetto import instantiate_annetto

from utils.annetto_utils import load_annetto_ontology


def join_unique_dir(dir: str, dir2: str):
    """ """

    path = os.path.join(dir, dir2)

    os.makedirs(path, exist_ok=True)

    if os.path.exists(path):
        i = 1
        while os.path.exists(f"{path}_{i}"):
            i += 1
        path = f"{path}_{i}"

    os.makedirs(path)
    return path


def main(ann_name: str, ann_path: str, output_ontology_filepath: str) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :param output_ontology_filepath: The path to the output ontology directory.
    :return: The path to the output ontology file.
    """
    if not ann_name:
        raise ValueError("ANN name is required.")
    if not ann_path:
        raise ValueError("ANN path is required.")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ANN path {ann_path} does not exist.")

    ann_pdf_files = glob.glob(f"{ann_path}/*.pdf")
    py_files = glob.glob(f"{ann_path}/*.py")
    onnx_files = glob.glob(f"{ann_path}/*.onnx")
    pb_files = glob.glob(f"{ann_path}/*.pb")

    # Check if any files were found
    if not ann_pdf_files and not py_files and not onnx_files and not pb_files:
        raise FileNotFoundError(f"No files found in {ann_path}.")

    # Extract text from PDF, if any
    if ann_pdf_files:
        for pdf_file in ann_pdf_files:
            extract_filter_pdf_to_json(pdf_file, ann_path)

    # Extract code (give file path, glob is processed in the function), if any
    if py_files or onnx_files or pb_files:
        process_code_file(ann_path)

    # Check if the output ontology path exists
    # If it doesn't, create it using the stable ontology
    # If it does, load the ontology from the given path
    if not os.path.exists(output_ontology_filepath):
        input_ontology = load_annetto_ontology(return_onto_from_release="stable")
    else:
        input_ontology = load_annetto_ontology(
            return_onto_from_path=output_ontology_filepath
        )
    # Instantiate Annett-o
    instantiate_annetto(ann_name, ann_path, input_ontology, output_ontology_filepath)


if __name__ == "__main__":
    main()
