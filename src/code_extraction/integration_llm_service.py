import unittest
import os
import glob
import pytest
from tests.deprecated.llm_service import init_engine , query_llm
import code_extractor
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json

ann_name = "alexnet"
code_files = glob.glob(f"data/{ann_name}/*.py")
pdf_file = f"data/{ann_name}/{ann_name}.pdf"

extract_filter_pdf_to_json(pdf_file)
code_extractor.process_code_file(code_files)

# now we have JSON files for both the papers & code
json_path = glob.glob(f"data/{ann_name}/*.json")

# call llm service
for count , j in enumerate(json_path):
    print(j)
    init_engine(j)

print("Query reached")

test_query1 = "Name each layer of this neural network sequentially, do not generalize internal layers and include modification and activation layers. Follow the JSON format that will be specified to you"
test_query2 = "Explain how AlexNet prevents overfitting from the technical details listed in the academic paper & the structure of the model given in python functions & code-related inputs."

query_llm(query=test_query2)

###################################
"""
NOTES on testing code & paper compatability:
- calling twice is not going to be viable, LLM fails to generate response every time
- chain of thought shows LLM is not able to balance the context between the two on different vector databases
- refactoring LLM_service allows for us to append to FAISS at each init_engine call, instead of creating a new vector db
- answers are much more accurate, may be the best solution for what we are working with
"""