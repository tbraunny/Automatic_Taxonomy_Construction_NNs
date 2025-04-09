# starting new:
#   - use ast to parse code
#    - look for specific functions in CNN (ex. 'Module' , '__init__' , 'forward')

import ast
import os

"""
DEPRECATED
"""

class CNN_parser(ast.NodeVisitor):
    def __init__(self):
        self.class_name = None
        self.layers = []
        self.forward_pass = []

    def visit_ClassDef(self , node):
        if any(
            (isinstance(base, ast.Attribute) and base.attr == 'Module') or
            (isinstance(base, ast.Name) and base.id == 'Module') for base in node.bases):

            self.class_name = node.name

            for body_item in node.body:
                if (isinstance(body_item , ast.FunctionDef) and body_item.name == '__init__'):
                    self.visit_init(body_item)
                elif (isinstance(body_item , ast.FunctionDef) and body_item.name == 'forward'):
                    self.visit_forward_pass(body_item)
            
    def visit_init(self , node):
        for info in node.body:
            if isinstance(info , ast.Assign) and isinstance(info.targets[0] , ast.Attribute):
                layer_name = info.targets[0].attr # grab layer name
                layer_type = None

                if isinstance(info.value , ast.Call): # if value of node is 
                    if isinstance(info.value.func , ast.Attribute):
                        layer_type = info.value.func.attr
                    elif isinstance(info.value.func , ast.Name):
                        layer_type = info.value.func.id

                    layer_values = [ast.dump(arg) for arg in info.value.args]

                    self.layers.append((layer_name , layer_type , layer_values))

    def visit_forward_pass(self , node):
        for info in node.body:
            if (isinstance(info , ast.Assign) and isinstance(info.value , ast.Call)):
                if isinstance(info.value.func , ast.Attribute) or isinstance(info.value.func , ast.Name):
                    self.forward_pass.append(ast.dump(info.value))

    def parse_code(self , code):
        tree = ast.parse(code)
        self.visit(tree)

    def embed_outputs(self , layers , forward):
        print("placeholder")

        embeddings = 0
        return embeddings

class CodeLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.code = ""

    def load(self):
        print("Loading Code...")
        with open(self.file_path , "r") as file:
            self.code = file.read()
        print("Finished loading code")

        return self.code

def py_to_txt(path , py_file):
    py_file_name , _ = os.path.splitext(py_file)
    txt_file = f"{py_file_name}.txt"

    try:
        with open(py_file, "r") as py_file, open(txt_file, "w") as txt_file:
            txt_file.write(py_file.read())
    finally:
        py_file.close()
        txt_file.close()

    return txt_file

def py_to_pdf(txt_file):
    pass

if __name__ == '__main__':
    path = "rag/"
    file = os.path.join(path , "code_parse_test.py")
    upload_path = "data/user_temp/"

    py_text = py_to_txt(upload_path , file)
    # code_loader = CodeLoader("test.py")
    # code = code_loader.load()

    # parser = CNN_parser()
    # parser.parse_code(code)

    # print("Architecture: ")
    # print("\tLayers: " , parser.layers)
    # print("\tForward Pass: " , parser.forward_pass)

    # embeddings = parser.embed_outputs(parser.layers , parser.forward_pass)

    # successful outputs, need to parse these outputs
    # comes in the form of the following:
    # Layers:
        # [('conv1', 'Conv2d', ['Constant(value=3)', 'Constant(value=20)', 'Constant(value=3)', 'Constant(value=1)']), ('conv2', 'Conv2d', ['Constant(value=20)', 'Constant(value=64)', 'Constant(value=3)', 'Constant(value=1)']), ('fc1', 'Linear', ['Constant(value=1600)', 'Constant(value=128)']), ('fc2', 'Linear', ['Constant(value=128)', 'Constant(value=2)']), ('bn1', 'BatchNorm2d', ['Constant(value=20)']), ('bn2', 'BatchNorm2d', ['Constant(value=64)']), ('dropout1', 'Dropout', ['Constant(value=0.5)']), ('dropout2', 'Dropout', ['Constant(value=0.25)'])]
    # Forward pass:
        #  ["Call(func=Attribute(value=Name(id='self', ctx=Load()), attr='conv1', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='F', ctx=Load()), attr='leaky_relu', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='F', ctx=Load()), attr='max_pool2d', ctx=Load()), args=[Name(id='x', ctx=Load()), Constant(value=2)], keywords=[])", "Call(func=Attribute(value=Name(id='self', ctx=Load()), attr='bn1', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='self', ctx=Load()), attr='dropout1', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='self', ctx=Load()), attr='conv2', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='F', ctx=Load()), attr='leaky_relu', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='F', ctx=Load()), attr='max_pool2d', ctx=Load()), args=[Name(id='x', ctx=Load()), Constant(value=2)], keywords=[])", "Call(func=Attribute(value=Name(id='self', ctx=Load()), attr='bn2', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='x', ctx=Load()), attr='view', ctx=Load()), args=[Call(func=Attribute(value=Name(id='x', ctx=Load()), attr='size', ctx=Load()), args=[Constant(value=0)], keywords=[]), UnaryOp(op=USub(), operand=Constant(value=1))], keywords=[])", "Call(func=Attribute(value=Name(id='self', ctx=Load()), attr='fc1', ctx=Load()), args=[Name(id='x', ctx=Load())], keywords=[])", "Call(func=Attribute(value=Name(id='F', ctx=Load()), attr='relu', ctx=Load()), args=[Name(id='x', ctx=Load(ant(value=64)', 'Constant(value=3)', 'Constant(value=1)']), ('fc1', 'Linear', ['Constant(value=1600)', 'Constant(value=128)']), ('fc2', 'Linear', ['Constant(value=128)', 'Constant(value=2)']), ('bn1', 'BatchNorm2d', ['Constant(value=20)']), ('bn2', 'BatchNorm2d', ['Constant(value=64)']), ('dropout1', 'Dropout', ['Constant(value=0.5)']), ('dropout2', 'Dropout', ['Constant(value=0.25)'])]