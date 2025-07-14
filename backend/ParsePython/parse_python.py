import ast
import tokenize
import os

class PythonParser(ast.NodeVisitor):
    def __init__(self):
        self.variables = []
        self.strings = []
        self.method_calls = []

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append(target.id)
        self.generic_visit(node)

    def visit_Constant(self, node):  # For Python 3.8+
        if isinstance(node.value, str):
            self.strings.append(node.value)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.method_calls.append(node.func.id)
        self.generic_visit(node)

# Parse
print(os.getcwd())
file_path = "backend/src/bert/inference/attack_surface_detection.py"
with open(file_path, "r") as f:
    tree = ast.parse(f.read())
    parser = PythonParser()
    parser.visit(tree)

print("Variables:", parser.variables)
# print("Strings:", parser.strings)
# print("Method Calls:", parser.method_calls)
