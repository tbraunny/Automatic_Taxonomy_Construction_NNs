import unittest
import os
import glob
import pytest
from utils.llm_service import init_engine , query_llm
import code_extractor
from src.pdf_extraction.extract_filter_pdf_to_json import extract_filter_pdf_to_json

ann_name = "alexnet"
code_files = glob.glob(f"data/{ann_name}/*.py")
pdf_file = f"data/{ann_name}/{ann_name}.pdf"

extract_filter_pdf_to_json(pdf_file)
code_extractor.process_code_file(code_files)

# now we have JSON files for both the papers & code
json_path = glob.glob(f"/data/{ann_name}/*.json")

# call llm service



# write out a test for code extraction
class TestCodeExtraction(unittest.TestCase):
    def test_visit_functions():
        print("test the visitation of all functions in code file")