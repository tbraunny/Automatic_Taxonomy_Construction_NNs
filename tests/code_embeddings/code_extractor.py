import ast
import json
import glob
import os

"""
Extract code from python files & convert into a JSON for langchain embeddings

Follows same format as PDF embeddings with 'page_content' & 'metadata' containing
all relevant information about the code.
"""
class CodeProcessor(ast.NodeVisitor):
    def __init__(self , code):
        self.code_lines = code.split("\n")
        self.sections = [] # classes / functions / global vars

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
        """

        class_code = self.extract_code_lines(node.lineno , node.end_lineno)
        class_section = {
            #"page_content": "\n".join(self.clean_code_lines(class_code)) , 
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


def process_code_file(filepath, count):
    """
    Traverse abstract syntax tree & dump relevant code into JSON
    """
    with open(filepath, "r") as f:
        code = f.read()

    tree = ast.parse(code)

    # for node in ast.walk(tree):
    #     for child in ast.iter_child_nodes(node):
    #         child.parent = node  # set reference nodes (ex. node.parent)
    
    processor = CodeProcessor(code)
    processor.visit(tree)

    output_file = filepath.replace(".py", f"_code{count}.json")
    with open(output_file, "w") as json_file:
        json.dump(processor.parse_code() , json_file , indent=3)
    
    print(f"JSONified code saved to {output_file}")


def main():
    ann_name = "alexnet"
    files = glob.glob(f"data/{ann_name}/*.py")
    print("File(s) found: " , files)

    for file_count, f in enumerate(files):
        process_code_file(f , file_count)


if __name__ == '__main__':
    main()