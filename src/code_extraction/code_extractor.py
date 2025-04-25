import os
import ast
import json
import glob
from utils.pytorch_extractor import extract_graph
from utils.tensorflow_extractor import extract_tf_graph
import logging
from datetime import datetime
import os
from utils.pb_extractor import PBExtractor
from utils.onnx_extractor import ONNXProgram
from utils.logger_util import get_logger
from utils.instantiate_dummy_args import create_dummy_value, instantiate_with_dummy_args
from utils.exception_utils import CodeExtractionError
from scripts.clean_repo import extract_dependencies
from scripts.unzip_clean_repo import unzip_and_clean_repo

# extra libraries for loading pytorch code into memory (avoids depenecy issues)
import torch
import torch.nn as nn
import torchvision
from torchvision import models as tmodels
import math
import numpy as np
import random
import scipy
import copy
import tqdm
import matplotlib
import keras
import argparse
import time
import datetime
import functools
import contextlib
import collections
import logging
import timm
import abc
import optree
import sklearn
import matplotlib
import ipdb

"""
Extract code from python files & convert into a JSON for langchain embeddings

Follows same format as PDF embeddings with 'page_content' & 'metadata' containing
all relevant information about the code.

NOTE: for how to run, see main()
"""

from utils.logger_util import get_logger
logger = get_logger("code_extraction")

class _CodeProcessor(ast.NodeVisitor):
    def __init__(self , code , file_path):
        self.code_lines = code.split("\n")
        self.sections = [] # classes / functions / global vars
        self.pytorch_model_graphs = {}
        self.model_name: str = None
        self.pytorch_module_names: list = [] # names of the different networks within a pytorch file
        self.module_namespace = self.extract_namespace(file_path)

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
            # PyTorch check
            if base.attr == "Module" and ( # check for nn.Module base class
                    (hasattr(base.value , "id") and base.value.id == "nn") or 
                    (hasattr(base.value , "value") and base.value.value.id == "torch" and base.value.attr == "nn")): 
                logging.info("PyTorch instantiation found")
                self.pytorch_module_names.append(node.name)
                mappings: dict = {}


                class_code = self.extract_code_lines(node.lineno , node.end_lineno)
                model_class = self.module_namespace.get(node.name , "\n".join(class_code))

                if not model_class:
                    logger.error(f"Could not find class {node.name} in namespace")
                    return
                
                model = instantiate_with_dummy_args(model_class) # might not need it
                if model is None and node.name in dir(tmodels):
                    try:
                        model = tmodels.get_model(node.name, weights='DEFAULT')
                    except Exception as e:
                        logging.warning(f"Could not load torchvision model {node.name}: {e}")
                
                if model:
                    self.model_name = node.name
                    graph = extract_graph(model)
                    self.pytorch_model_graphs[node.name] = graph

                # class_code = self.extract_code_lines(node.lineno , node.end_lineno) # fetch code associated w class
                # exec("\n".join(class_code) , globals() , mappings) # load code into memory
                # model_class = mappings.get(node.name)

                # if model_class:
                #     model = instantiate_with_dummy_args(model_class)

                #     if model is None and node.name in dir(tmodels):
                #         try:
                #             model = tmodels.get_model(node.name, weights='DEFAULT')
                #         except Exception as e:
                #             logging.warning(f"Could not load torchvision model {node.name}: {e}")
                    
                #     if model:
                #         self.model_name = node.name
                #         self.pytorch_graph = extract_graph(model)              

            # Tensorflow check
            elif hasattr(base, "attr") and base.attr == "Model":
                if (
                    (hasattr(base.value, "id") and base.value.id in ("keras", "tf")) or
                    (hasattr(base.value, "attr") and base.value.attr == "keras" and hasattr(base.value, "value") and base.value.value.id == "tf")
                ):
                    logging.info("TensorFlow/Keras instantiation found")
                    mappings: dict = {}

                    class_code = self.extract_code_lines(node.lineno , node.end_lineno)
                    exec("\n".join(class_code), globals(), mappings)
                    model_class = mappings.get(node.name)
                    model = model_class()

                    self.model_name = node.name
                    self.tf_graph: dict = extract_tf_graph(node)

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
    
    def extract_namespace(self , filepath: str , code):
        with open(filepath, 'r') as f:
            code = f.read()
        namespace = {
            # '__name__': '__main__',  # hope this helps (might have to take out)
            # '__file__': filepath,
            # '__package__': None,
            # '__builtins__': __builtins__,
        }

        #compiled_code = _strip_relative_imports(code)
        try:
            exec(code, namespace)
        except Exception as e:
            raise CodeExtractionError(
                message="PyTorch code could not be loaded If file contains relative imports, try uploading entire repository as ZIP",
                code="PYTORCH_IMPORTS",
                context={"datatype": 405 , "property": "PYTORCH"}
            )
        return namespace     

class CodeExtractor():
    def __init__(self):
        self.pytorch_module_names:list = []
        self.pytorch_present: bool = False

    def save_json(self , output_file: str , content: dict):
        with open(output_file, "w") as json_file:
            json.dump(content , json_file , indent=3)
                
        logger.info(f"JSON successfully saved to {output_file}")

    def process_code_file(self , file_path) -> list:
        """
        Traverse abstract syntax tree & dump relevant code into JSON. Given model directory,
        automatically detects pytorch & handles both ONNX & TensorFlow files with ANNETT-O
        instantiation.

        :param file_path: Directory in which code files may be present (eg. data/{ann_name})
        :return List of the pytorch module names
        """
        try:
            code_files_present: bool = False
            file_path  = os.path.normpath(file_path)
            py_files = glob.glob(f"{file_path}/**/*.py" , recursive=True)
            onnx_files = glob.glob(f"{file_path}/**/*.onnx" , recursive=True)
            pb_files = glob.glob(f"{file_path}/**/*.pb" , recursive=True)
            zip_file = f"{file_path}/*.zip"


            if onnx_files:
                logger.info(f"ONNX file(s) detected: {onnx_files}")
                code_files_present = True
                for count , file in enumerate(onnx_files):
                    logger.info(f"Parsing ONNX file {file}...")
                    #output_json = file.replace(".onnx" , f"onnx_{count}.json")
                    onnx_graph: dict = ONNXProgram().compute_graph_extraction(file)
                    self.save_json(file.replace(".onnx" , f"_onnx_{count}.json") , onnx_graph)
            if pb_files:
                logger.info(f"TensorFlow file(s) detected: {pb_files}")
                code_files_present = True
                for count , file in enumerate(pb_files):
                    logger.info(f"Parsing TensorFlow file {file}...")
                    #output_json = file.replace(".pb" , f"_pbcode_{count}.json")
                    pb_graph = PBExtractor.extract_compute_graph(file)
                    self.save_json(file.replace(".pb" , f"_pb_{count}.json") , pb_graph)
            dirty_py_files: list = py_files
            if zip_file: # unzip & clean the uploaded repo
                user_repo_path = os.path.join(ann_path, "user_repo")

                if not py_files:
                    raise CodeExtractionError(
                        message="Ensure the model py file is uploaded in addition to the ZIP file",
                        code="TARGET_MODEL_FILE_MISSING",
                        context={"datatype": 406 , "property": "TARGET_MODEL_FILE"}
                    )
                unzip_and_clean_repo(zip_file , user_repo_path)
                
                for py_file in dirty_py_files:
                    py_name = os.path.basename(py_file)
                    found_clean_file = None
                    for root, dirs, files in os.walk(user_repo_path):
                        if py_name in files:
                            found_clean_file = os.path.join(root, py_name)
                            break

                    if found_clean_file:
                        py_files.append(found_clean_file)
                        extract_dependencies(user_repo_path, found_clean_file)
                        ann_path = user_repo_path
                    else:
                        raise CodeExtractionError(
                            message=f"File '{py_name}' not found in uploaded repository.",
                            code="CLEANED_MODEL_FILE_MISSING",
                            context={"datatype": 407 , "property": "CLEAN_REPO"}
                        )
            if py_files: # informational
                logger.info(f"Python file(s) detected: {py_files}")
                code_files_present = True
                for count , file in enumerate(py_files):
                    logger.info(f"Parsing python file {file}...")
                    code = 0
                    with open(file , "r") as f:
                        code = f.read()
                    tree = ast.parse(code)
                    output_file = file.replace(".py", f"_code_doc_{count}.json")
                    processor = _CodeProcessor(code , file)

                    # for node in ast.walk(tree): # track nodes
                    #     for child in ast.iter_child_nodes(node):
                    #         child.parent = node  # set reference nodes (ex. node.parent)
                    processor.visit(tree)

                    if not processor.model_name:
                        base = os.path.basename(file)
                        processor.model_name = os.path.splitext(base)[0]
                    if processor.pytorch_model_graphs: # symbolic graph dictionary
                        logger.info(f"PyTorch code found within file {file}")
                        self.pytorch_module_names = processor.pytorch_module_names

                        for model_name , graph in processor.pytorch_model_graphs.items(): # save each class individually
                            self.save_json(file.replace(".py" , f"_{model_name}_torch_{count}.json") , graph)
                        #self.save_json(file.replace(".py", f"_code_torch_{count}.json") , processor.pytorch_graph)
                    elif processor.tf_graph:
                        logger.info(f"TensorFlow code found within file {file}")
                        self.save_json(file.replace(".py" , f"_code_tf_{count}.json") , processor.tf_graph)
                    else:
                        logger.info(f"Model name '{processor.model_name}' is not PyTorch or TensorFlow")
                        self.pytorch_present = False

                    # regular code dictionary for RAG
                    output = processor.parse_code()
                    self.save_json(output_file , output)
            if not code_files_present:
                logger.error(f"No code file(s) of any type found")

                raise CodeExtractionError(
                    message=f"No code/model files found in directory {file_path}",
                    code="FILES_NOT_FOUND",
                    context={"code_extraction": "500", "property": FileNotFoundError},
                )

            return processor.pytorch_module_names
                
        except Exception as e:
            logger.error(f"Error processing code file(s), {e}" , exc_info=True)
            return

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

def main():
    ann_name = "bart_ae"
    filepath = f"data/more_papers/{ann_name}"
    #logger.info(f"File(s) found: {filepath}")
    
    # simply provide the file path that may contain the related network code files
    processor = CodeExtractor()
    processor.process_code_file(filepath)
    print(processor.pytorch_module_names) # PYTORCH MODULES NAMES


if __name__ == '__main__':
    main()