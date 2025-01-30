import ast
import json
import glob
import os

class CodeProcessor(ast.NodeVisitor):
    def __init__(self, code , filepath): # initialize lists
        self.metadata = []
        self.classes = []
        self.functions = []
        self.assignments = []
        self.code_lines = code.split("\n")
        self.metadata = {
            "file_name": os.path.basename(filepath),
            "file_size": os.path.getsize(filepath),  # size in bytes
            "num_lines": len(self.code_lines),
            "num_functions": 0,  
            "num_classes": 0 ,  
            "imports": []
        }

    def visit_Import(self, node): # visit import statements
        for alias in node.names:
            self.metadata["imports"].append(alias.name)

    def visit_ImportFrom(self, node): # visit from ... import statements
        module_name = node.module
        for alias in node.names:
            self.metadata["imports"].append(f"{module_name}.{alias.name}")

    def capture_variable_assignments(self, node):
        # capture all variable assignments to encopmass hyperparameters
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value = ast.unparse(node.value)
                    self.assignments.append({target.id: value})

    def visit_ClassDef(self, node): # extracts class names & methods (includes function calls within methods)
        self.metadata["num_classes"] += 1
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
        self.metadata["num_functions"] += 1
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
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name): # capture variable assignments
                value = ast.unparse(node.value)  # get value
                self.assignments.append({target.id: value})

        self.generic_visit(node)

    def parse_code(self): # structure the JSON
        return {
            "metadata": self.metadata ,
            "classes": self.classes ,
            "functions": self.functions ,
            "assignments": self.assignments
        }

def process_code_file(filepath , count):
    with open(filepath, "r") as f:
        code = f.read()

    processor = CodeProcessor(code , filepath)
    tree = ast.parse(code) # load code tree
    processor.visit(tree) # traverse nodes

    output_file = filepath.replace(".py", f"_processed_{count}.json")
    with open(output_file, "w") as json_file:
        json.dump(processor.parse_code(), json_file, indent=3)
    
    print(f"Processed code saved to {output_file}")
    return processor.parse_code()

def main():
    ann_name = "alexnet"
    files = glob.glob(f"/home/richw/tom/ATCNN/data/{ann_name}/*.py") # accept all python files listed under a neural network folder

    file_count = 0
    for f in files:
        process_code_file(f , file_count)
        file_count += 1

if __name__ == '__main__':
    main()