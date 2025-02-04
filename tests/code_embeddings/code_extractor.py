import ast
import json
import glob
import os


class CodeProcessor(ast.NodeVisitor):
    def __init__(self , code):
        self.code_lines = code.split("\n")
        self.classes = {}
        self.functions = {}


    def visit_ClassDef(self, node):
        """
        Visits nodes that are a class traversing down tree from given node
        """

        class_info = {"methods": {}}

        for body_item in node.body:
            if isinstance(body_item , ast.FunctionDef):
                method_name = body_item.name
                parameters = [arg.arg for arg in body_item.args.args]
                function_code = self.extract_code_lines(body_item.lineno , body_item.end_lineno) # capture strictly code within scope
                class_info["methods"][method_name] = {
                    "parameters": parameters ,
                    "code": function_code
                }

        self.classes[node.name] = class_info
        self.generic_visit(node) # travel to next node


    def visit_FunctionDef(self , node):
        """
        Visit nodes defined as function declarations from a given node, 
        capture the params & code
        """
        
        if isinstance(node.parent, ast.Module): # ensure top-level function
            function_code = self.extract_code_lines(node.lineno, node.end_lineno)
            self.functions[node.name] = {
                "parameters": [arg.arg for arg in node.args.args],
                "code": function_code
            }


    def extract_code_lines(self , start , end):
        """
        Currently keeps comments in when extracting lines, could be useful, could be changed later
        """
        return self.code_lines[start - 1:end]  # line counts start at 1, adjust from 0
    


    def parse_code(self):
        """
        Return contents of code to JSON dump
        """

        return {
            "classes": self.classes,
            "functions": self.functions
        }


def process_code_file(filepath, count):
    with open(filepath, "r") as f:
        code = f.read()

    tree = ast.parse(code)
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # set reference
    
    processor = CodeProcessor(code)
    processor.visit(tree)

    output_file = filepath.replace(".py", f"{count}_code.json")
    with open(output_file, "w") as json_file:
        json.dump(processor.parse_code() , json_file , indent=3)
    
    print(f"JSONified code saved to {output_file}")


def main():
    ann_name = "alexnet"
    files = glob.glob(f"/home/richw/tom/ATCNN/data/{ann_name}/*.py")

    for file_count, f in enumerate(files):
        process_code_file(f , file_count)


if __name__ == '__main__':
    main()