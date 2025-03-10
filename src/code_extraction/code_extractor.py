import ast
import json
import glob
from pytorchgraphextraction import extract_graph
import logging
from datetime import datetime
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

NOTE: if pytorch code detected, code file is passed to Chase's symbolic extraction
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
            if base.attr == "Module" and base.value.id == "nn": # check for nn.Module base class
                mappings: dict = {}

                class_code = self.extract_code_lines(node.lineno , node.end_lineno) # fetch code associated w class
                exec("\n".join(class_code) , globals() , mappings) # load code into memory
                model_class = mappings.get(node.name)
                model = model_class()
                model = tmodels.get_model(node.name , weights='DEFAULT') # preprocessing?

                self.pytorch_graph = extract_graph(model)                
        
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

def process_code_file(files):
    """
    Traverse abstract syntax tree & dump relevant code into JSON

    :param files: Directory in which code files may be present (data/ann_name/*.py)
    :return None: JSON files saved to same directory given
    """
    try:
        for count , file in enumerate(files):
            with open(file , "r") as f:
                code = f.read()
            tree = ast.parse(code)
            output_file = file.replace(".py", f"_code_{count}.json")

            # for node in ast.walk(tree): # track nodes
            #     for child in ast.iter_child_nodes(node):
            #         child.parent = node  # set reference nodes (ex. node.parent)
            
            processor = CodeProcessor(code)
            processor.visit(tree)
            output: dict = {}

            if processor.pytorch_graph: # symbolic graph dictionary
                output = processor.pytorch_graph
                output_file = file.replace(".py", f"_code_torch_{count}.json")
            else: # regular code dictionary
                output = processor.parse_code()

            with open(output_file, "w") as json_file:
                json.dump(output , json_file , indent=3)
            
            print(f"JSONified code saved to {output_file}")
            logger.info(f"JSON successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"Error processing code files, {e}")


def main():
    ann_name = "alexnet"
    files = glob.glob(f"data/{ann_name}/*.py")
    print("File(s) found: " , files)

    process_code_file(files)


if __name__ == '__main__':
    main()