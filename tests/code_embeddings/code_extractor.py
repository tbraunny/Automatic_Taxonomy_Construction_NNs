import ast
import json
import glob
import os

class CodeProcessor(ast.NodeVisitor):
    def __init__(self, code): # initialize lists
        self.classes = []
        self.functions = []
        #self.imports = []
        self.code_lines = code.split("\n")

    def visit_Import(self, node): # visit import statements
        for alias in node.names:
            self.imports.append(alias.name)

    def visit_ImportFrom(self, node): # visit from ... import statements
        module_name = node.module
        for alias in node.names:
            self.imports.append(f"{module_name}.{alias.name}")

    def visit_ClassDef(self, node): # extracts class names & methods (includes function calls within methods)
        class_info = {
            "name": node.name,
            "methods": {},
            "init_function_calls": []
        }

        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef):
                method_name = body_item.name
                parameters = [arg.arg for arg in body_item.args.args]

                if method_name == "__init__":

                    # extract function calls within __init__
                    function_calls = self.extract_function_calls(body_item)
                    class_info["init_function_calls"] = function_calls

                class_info["methods"][method_name] = {"parameters": parameters}

        self.classes.append(class_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node): # fetch function names & params
        function_name = node.name
        parameters = [arg.arg for arg in node.args.args]
        self.functions.append({"name": function_name, "parameters": parameters})

    def find_end_line(self, node): # find scope release in a function
        last_line = node.lineno
        for child in ast.iter_child_nodes(node): # recurse down tree until end of child's code block is found
            if hasattr(child, "lineno"):
                last_line = max(last_line, self.find_end_line(child))
        return last_line

    def extract_function_calls(self, node): # fetch function calls & params within methods
        function_calls = []
        for child in ast.walk(node): # walk the tree
            if isinstance(child, ast.Call):
                func_name = self.get_func_name(child)
                args = [ast.unparse(arg) for arg in child.args]  # extract arguments as strings
                function_calls.append({"function": func_name, "arguments": args}) # append to list
        return function_calls

    def get_func_name(self, node): # fetch function name from a function call
        if isinstance(node.func, ast.Attribute):  # handles 'self' function calls
            return node.func.attr
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return "unknown"

    def parse_code(self): # structure the JSON
        return {
            #"imports": self.imports,
            "classes": self.classes,
            "functions": self.functions
        }

def process_code_file(filepath):
    with open(filepath, "r") as f:
        code = f.read()

    processor = CodeProcessor(code)
    tree = ast.parse(code) # load code tree
    processor.visit(tree) # traverse nodes

    output_file = filepath.replace(".py", "_processed.json")
    with open(output_file, "w") as json_file:
        json.dump(processor.parse_code(), json_file, indent=3)
    
    print(f"Processed code saved to {output_file}")
    return processor.parse_code()

def main():
    ann_name = "alexnet"
    files = glob.glob(f"/home/richw/tom/ATCNN/data/{ann_name}/*.py") # accept all python files listed under a neural network folder

    for f in files:
        process_code_file(f)

if __name__ == '__main__':
    main()