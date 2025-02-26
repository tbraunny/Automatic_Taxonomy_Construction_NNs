import unittest
import ast
from code_extractor import CodeProcessor  

class TestCodeProcessor(unittest.TestCase):
    """
    Test the proper organization of code according to the modeule
    that it belongs to (global vars, class, function)
    """

    def test_global_variable_extraction(self):
        code = """x = 10\ny = 20\nif __name__ == '__main__':\n\tz = 30"""
        processor = CodeProcessor(code)
        tree = ast.parse(code)
        processor.visit(tree)
        
        global_vars = next((s for s in processor.sections if s["metadata"]["section_header"] == "Global Variables"), None)
        global_other = next((s for s in processor.sections if s["metadata"]["section_header"] == "Global Other") , None)
        self.assertIsNotNone(global_vars)
        self.assertIn("x = 10" , global_vars["page_content"])
        self.assertIn("y = 20" , global_vars["page_content"])
        self.assertIn("z = 30" , global_other["page_content"])
    
    def test_function_extraction(self):
        code = """def my_function():\n\treturn 42"""
        processor = CodeProcessor(code)
        tree = ast.parse(code)
        processor.visit(tree)
        
        function_section = next((s for s in processor.sections if s["metadata"]["section_header"] == "my_function"), None)
        self.assertIsNotNone(function_section)
        self.assertIn("def my_function():", function_section["page_content"])
    
    def test_class_extraction(self):
        code = """class MyClass:\n\tdef method_one(self):\n\t\tpass\n\tdef method_two(self):\n\t\tpass"""
        processor = CodeProcessor(code)
        tree = ast.parse(code)
        processor.visit(tree)
        
        class_section = next((s for s in processor.sections if s["metadata"]["section_header"] == "MyClass"), None)
        self.assertIsNotNone(class_section)
        self.assertIn("Functions: method_one, method_two", class_section["page_content"])

    def test_clean_code_lines(self):
        code_lines = ["    def my_function():", "        return 42", ""]
        processor = CodeProcessor("")
        cleaned_lines = processor.clean_code_lines(code_lines)
        self.assertEqual(cleaned_lines, ["\tdef my_function():", "\t\treturn 42"])
        
if __name__ == '__main__':
    unittest.main()