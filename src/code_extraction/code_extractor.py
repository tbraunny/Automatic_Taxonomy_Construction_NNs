import ast
import json
import glob
from utils.pytorch_extractor import extract_graph
import logging
from datetime import datetime
import os
from utils.onnx_db import check_onnx
from utils.pb_extractor import PBExtractor
from utils.onnx_extractor import ONNXProgram
#from tests.deprecated.pt_extractor import PTExtractor
import os

# extra libraries for loading pytorch code into memory (avoids depenecy issues)
import torch
import torch.nn as nn
import torchvision
from torchvision import models as tmodels

"""
Extract code from python files & convert into a JSON for langchain embeddings

Follows same format as PDF embeddings with 'page_content' & 'metadata' containing
all relevant information about the code.

NOTE: for how to run, see main()
"""

log_dir = "logs"
log_file = os.path.join(log_dir , f"code_extraction_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log")
os.makedirs(log_dir , exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Write to file
    ],
    force=True,
)
logger = logging.getLogger(__name__)

class CodeProcessor(ast.NodeVisitor):
    def __init__(self , code):
        self.code_lines = code.split("\n")
        self.sections = [] # classes / functions / global vars
        self.pytorch_graph = []
        self.model_name: str = None

    def visit_Module(self, node):
        """
        Capture global statements in case of hyperparameters or important info
        specified at the module level
        """
        global_vars = []
        global_other = []

        for stmt in node.body:
            if isinstance(stmt, ast.Assign): # check for assignment in module node
                code_lines = self.extract_code_lines(stmt.lineno , stmt.end_lineno)
                global_vars.extend(code_lines)
            else:
                code_lines = self.extract_code_lines(stmt.lineno , stmt.end_lineno)
                global_other.extend(code_lines)

        global_vars_section = {
            "page_content": "\n".join(self.clean_code_lines(global_vars)),
            "metadata": {
                "section_header": "Global Variables",
                "type": "python global"
            }
        }
        self.sections.append(global_vars_section)

        global_other_section = {
            "page_content": "\n".join(self.clean_code_lines(global_other)) , 
            "metadata": {
                "section_header": "Global Other",
                "type": "python global"
            }
        }
        self.sections.append(global_other_section)
        self.generic_visit(node) # travel to next node


    def visit_ClassDef(self, node):
        """
        Visits nodes that are a class traversing down tree from given node
        Also checks for class that instantiates PyTorch model
        """
        for base in node.bases:
            if base.attr == "Module" and ( # check for nn.Module base class
                    (hasattr(base.value , "id") and base.value.id == "nn") or 
                    (hasattr(base.value , "value") and base.value.value.id == "torch" and base.value.attr == "nn")): 
                logging.info("PyTorch instantiation found")
                mappings: dict = {}

                class_code = self.extract_code_lines(node.lineno , node.end_lineno) # fetch code associated w class
                exec("\n".join(class_code) , globals() , mappings) # load code into memory
                model_class = mappings.get(node.name)
                model = model_class()
                model = tmodels.get_model(node.name , weights='DEFAULT') # preprocessing?
                self.model_name = node.name

                self.pytorch_graph = json.loads(extract_graph(model)) # extract_graph returns json.dumps                
        
        class_section = {
            #"page_content": "\n".join(self.clean_code_lines(class_code)) , # clean up code lines
            "page_content" : "Functions: " + (", ".join([f"{func.name}" for func in node.body if isinstance(func , ast.FunctionDef)])) , 
            "metadata": {
                "section_header": node.name , 
                "type": "python class"
            }
        }
        self.sections.append(class_section)
        self.generic_visit(node) # travel to next node


    def visit_FunctionDef(self , node):
        """
        Visit nodes defined as function declarations from a given node, 
        capture the params & code
        """

        function_code = self.extract_code_lines(node.lineno , node.end_lineno)
        function_section = {
            "page_content": "\n".join(self.clean_code_lines(function_code)) , 
            "metadata": {
                "section_header": node.name ,
                "type": "python function"
            }
        }
        self.sections.append(function_section)


    def clean_code_lines(self, lines):
        """
        Clean up lines to remove unnecessary blank lines while preserving indentation
        """
        # Remove lines that are completely blank or contain only whitespace
        cleaned_lines = []
        for line in lines:
            # Remove blank lines and trailing spaces
            line = line.rstrip()  # Remove trailing spaces
            if line.strip():  # Only keep non-blank lines
                # Replace leading spaces with tabs
                line = line.replace("    ", "\t")
                cleaned_lines.append(line)
        return cleaned_lines
        #return [line.rstrip() for line in lines if line.strip()] #if the \t in the json gives the model trouble


    def extract_code_lines(self , start , end):
        """
        Currently keeps comments in when extracting lines, could be useful, could be changed later
        """
        return self.code_lines[start - 1:end]  # line counts start at 1, adjust from 0


    def parse_code(self):
        """
        Return contents of code for JSON dump
        """
        return self.sections


def check_pytorch(tree: ast.Module) -> bool:
    """
    DEPRECATED
    Check if code file utilizes pytorch. If so, flag true
    
    :param tree: tree returned from ast.parse
    :return True if file contains torch-related imports
    """
    try:
        for node in ast.walk(tree):
            if isinstance(node , ast.Import) or isinstance(node , ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "torch":
                        logger.info(f"Detected PyTorch input from 'import', {alias}")
                        return True
                    elif node.module and node.module.startswith("torch"):
                        logger.info(f"Detected PyTorch input from 'from', {alias}")
                        return True
        return False
    except Exception as e:
        logger.error(f"Check for PyTorch failed, {e}")
        return False


def save_json(output_file: str , content: dict):
    with open(output_file, "w") as json_file:
        json.dump(content , json_file , indent=3)
            
    logger.info(f"JSON successfully saved to {output_file}")

def process_code_file(file_path):
    """
    Traverse abstract syntax tree & dump relevant code into JSON. Given model directory,
    automatically detects pytorch & handles both ONNX & TensorFlow files with ANNETT-O
    instantiation.

    :param file_path: Directory in which code files may be present (eg. data/{ann_name})
    :return None: JSON files saved to same directory given
    """
    try:
        py_files = glob.glob(f"{file_path}/*.py")
        #pt_files = glob.glob(f"{file_path}/*.pt") # still working on it
        onnx_files = glob.glob(f"{file_path}/*.onnx")
        pb_files = glob.glob(f"{file_path}/*.pb")

        if onnx_files:
            logger.info(f"ONNX files detected: {onnx_files}")
            for count , file in enumerate(onnx_files):
                logger.info(f"Parsing ONNX file {file}...")
                output_json = file.replace(".onnx" , f"onnx_{count}.json")
                ONNXProgram().extract_properties(file , savePath=output_json) # how should we run the onnx extractor
                print("ONNX file parsed & saved to: " , output_json)
        if pb_files:
            logger.info(f"TensorFlow files detected: {pb_files}")
            for count , file in enumerate(pb_files):
                logger.info(f"Parsing TensorFlow file {file}...")
                output_json = file.replace(".pb" , f"_pbcode_{count}.json")
                PBExtractor.extract_compute_graph(file , output_json)
        # if pt_files:
        #     logger.info(f"PyTorch weights & biases detected: {pt_files}")
        #     for count , file in enumerate(pt_files):
        #         logger.info(f"Parsing PyTorch file {file}...")
        #         output_json = file.replace(".pt" , f"_ptcode{count}.json")
        #         PTExtractor.extract_compute_graph(file , output_json)
        if not py_files:
            print(file_path)
            print(py_files)
            logger.info("No Python files found in directory")

        for count , file in enumerate(py_files):
            logger.info(f"Parsing python file {file}...")
            with open(file , "r") as f:
                code = f.read()
            tree = ast.parse(code)
            output_file = file.replace(".py", f"_code_{count}.json")

            # for node in ast.walk(tree): # track nodes
            #     for child in ast.iter_child_nodes(node):
            #         child.parent = node  # set reference nodes (ex. node.parent)
            
            processor = CodeProcessor(code)
            processor.visit(tree)

            if not processor.model_name:
                base = os.path.basename(file)
                processor.model_name = os.path.splitext(base)[0]
            #onnx_model = check_onnx(processor.model_name) # check for onnx model

            if processor.pytorch_graph: # symbolic graph dictionary
                logger.info(f"PyTorch code found within file {file}")
                content = processor.pytorch_graph
                save_json(file.replace(".py", f"_code_torch_{count}.json") , content)
            # elif onnx_model: # check for model in onnx
            #     logger.info(f"Model name '{onnx_model}' found within ONNX database")
            #     # instantiate it
            else:
                logger.info(f"Model name '{processor.model_name}' is not PyTorch or an instance in the ONNX database")

            # regular code dictionary for RAG
            output = processor.parse_code()
            save_json(output_file , output)
            
    except Exception as e:
        print(e)
        logger.error(f"Error processing code files, {e}")


def main():
    ann_name = "alexnet"
    filepath = f"data/{ann_name}"
    #logger.info(f"File(s) found: {filepath}")

    # simply provide the file path that may contain the related network code files
    process_code_file(filepath)


if __name__ == '__main__':
    main()