
import os
import glob
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
from utils.exception_utils import CodeExtractionError, PDFError
import warnings
from typing import List
from src.taxonomy.llm_generate_criteria import llm_create_taxonomy

from utils.logger_util import get_logger

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import logging
logging.getLogger("faiss").setLevel(logging.ERROR)


logger = get_logger("main", max_logs=3)

def remove_ann_config_from_user_owl(ann_name: str) -> None:
    """
    Removes the ANN configuration from the Annett-o ontology.
    Written specifically for user-appended ontologies.
    Saves the updated owl file to it's original location.

    :param ann_name: The name of the ANN.
    """
    if not ann_name or not isinstance(ann_name, str):
        logger.error("ANN name is required.")
        raise ValueError("ANN name is required.")
    
    owl_file = C.ONTOLOGY.USER_OWL_FILENAME
    ontology = load_annetto_ontology(return_onto_from_path=owl_file)
    with ontology:
        from utils.owl_utils import get_class_instances
        ann_instances = get_class_instances(ontology.ANNConfiguration)
        for ann_instance in ann_instances:
            ann_instance_name = ann_instance.name
            stripped_ann_instance_name = ann_instance_name.split('_', 1)[1] if '_' in ann_instance_name else ann_instance_name
            if stripped_ann_instance_name == ann_name:
                delete_ann_configuration(ontology, ann_instance_name)
                break
        save_ontology(ontology, owl_file)
        logger.info(f"Removed ANN configuration for {ann_name} from user owl file.")

def main(ann_name: str, ann_path: str, use_user_owl: bool = True, test_input_ontology_filepath: str = None, test_output_ontology_filepath: str = None) -> str:
    """
    Main function to run the Annett-o pipeline.

    :param ann_name: The name of the ANN.
    :param ann_path: The path to the directory containing the ANN files.
    :param output_ontology_filepath: The path to save the output ontology file, if you pass this, 
                                     it will overwrite the user onto file (lukas pls dont pass anything for this param, use the user_owl bool)
    :param use_user_owl: Whether to use the user-appended ontology or the pre-made stable ontology.
    """
    if ann_path.endswith(".zip"):
        from scripts.unzip_clean_repo import unzip_and_clean_repo
        new_ann_path = os.path.join(C.ONTOLOGY.USER_OWL_FILENAME, ann_name)
        try:
            unzip_and_clean_repo(ann_path, new_ann_path)
        except Exception as e:
            logger.error(f"Error unzipping the file: {e}")
            raise
        ann_path = new_ann_path
        logger.info(f"Unzipped {ann_path} to {new_ann_path}.")

    # if a annpath has a zip file, we need to unzip it
    # then we need to match the py files in to the same file in the zip dir
    # clean that zip dir
    # need to come up with how we pass this to Tom

    if not os.path.isdir(ann_path):
        logger.error("ANN path must be a directory.")
        raise ValueError("ANN path must be a directory.")

    # Check if the ann_path has pdfs
    ann_pdfs = glob.glob(os.path.join(ann_path,"*.pdf"))
    if not ann_pdfs:
        logger.error(f"No PDF files found in {ann_path}.")
        raise PDFError(
        message="No PDFs provided. Please provide a PDF.",
        code="PDF_NOT_FOUND",
        context={"datatype": "422", "property": "pdf"})
    # If ann_path has multiple pdfs, use the first one and log a warning
    # NOTE: Can only handle one pdf for now. Assumes a single ANN can't have multiple papers 
    if len(ann_pdfs) > 1:
        logger.warning(f"Multiple PDF files found in {ann_path}. Using the first one.")
        ann_pdf = ann_pdfs[0]
    else:
        ann_pdf = ann_pdfs[0]

    # TODO: Use llm query to verify if the pdf is about a NN architecture

    # Extract text from PDF
    extract_filter_pdf_to_json(ann_pdf, ann_path)
    logger.info(f"Extracted text from {ann_pdf} to JSON.")

    # Extract code (give file path, glob is processed in the function), if any
    pytorch_module_names: List[str] = []
    process_code = CodeExtractor()
    process_code.process_code_file(ann_path)
    pytorch_module_names = process_code.pytorch_module_names
    logger.info(f"Extracted code from to JSON.")

    # insert model into db 
    # NOTE: This is for Chase
    db_runner = DBUtils()
    model_id: int = db_runner.insert_model_components(ann_path) # returns id of inserted model
    paper_id: int = db_runner.insert_papers(ann_path)
    translation_id: int = db_runner.model_to_paper(model_id, paper_id)

    if test_input_ontology_filepath:
        logger.info("Using parameter passed ontology.")
        input_ontology = load_annetto_ontology(
            return_onto_from_path=output_ontology_filepath
        )
    elif not use_user_owl:
        logger.info("Using the stable ontology.")
        input_ontology = load_annetto_ontology(return_onto_from_release="stable")
    else:
        logger.info("Using the user ontology.")
        input_ontology = load_annetto_ontology(
            return_onto_from_path=C.ONTOLOGY.USER_OWL_FILENAME
        )

    if test_output_ontology_filepath:
        output_ontology_filepath = test_output_ontology_filepath
    else:
        output_ontology_filepath = C.ONTOLOGY.USER_OWL_FILENAME

    # Initialize Annett-o with new classes and properties
    initialize_annetto(input_ontology, logger)
    logger.info("Initialized Annett-o ontology.")

    # Instantiate Annett-o
    instantiate_annetto(ann_name, ann_path, input_ontology, output_ontology_filepath, pytorch_module_names)
    logger.info("Instantiated Annett-o ontology.")

    # Define split criteria via llm
    ontology = load_annetto_ontology(return_onto_from_path=output_ontology_filepath)
    # thecriteria = llm_create_taxonomy('What would you say is the taxonomy that represents all neural network?', ontology)
    # taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    # format='json'
    # topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(format=format,faceted=True)
    
    # # Create faceted taxonomy as df
    # df = create_tabular_view_from_faceted_taxonomy(taxonomy_str=json.dumps(serialize(facetedTaxonomy)), format=format)
    # df.to_csv("./data/taxonomy/faceted/generic/generic_taxonomy.csv")


if __name__ == "__main__":
    ann_name = "inceptionv3_cnn"
    user_path = "data/userinput"
    user_ann_path = os.path.join(user_path, ann_name)
    os.makedirs(user_ann_path, exist_ok=True)
    main(ann_name, user_ann_path, use_user_owl=False)
    remove_ann_config_from_user_owl(ann_name)