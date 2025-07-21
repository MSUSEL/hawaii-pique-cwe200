import ast
import json
import os

class PythonParser(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}
        self.strings = {}
        self.method_calls = {}
        self.method_code = {}
        self.current_method = "global"

    def visit_FunctionDef(self, node):
        method_name = node.name
        self.current_method = method_name
        code = ast.unparse(node) if hasattr(ast, 'unparse') else "<code unavailable>"
        self.method_code[method_name] = code
        self.generic_visit(node)
        self.current_method = "global"

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_info = self.variables.get(var_name, {"name": var_name, "type": "unknown", "methods": []})
                if self.current_method not in var_info["methods"]:
                    var_info["methods"].append(self.current_method)
                self.variables[var_name] = var_info
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value
            str_info = self.strings.get(value, {"name": value, "methods": []})
            if self.current_method not in str_info["methods"]:
                str_info["methods"].append(self.current_method)
            self.strings[value] = str_info
        self.generic_visit(node)

    def visit_Call(self, node):
        call_name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', "<unknown>")
        call_info = self.method_calls.get(call_name, {"name": call_name, "methods": []})
        if self.current_method not in call_info["methods"]:
            call_info["methods"].append(self.current_method)
        self.method_calls[call_name] = call_info
        self.generic_visit(node)

def parse_python_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    parser = PythonParser()
    parser.visit(tree)

    return {
        "filename": os.path.basename(file_path),
        "variables": list(parser.variables.values()),
        "strings": list(parser.strings.values()),
        "methodCodeMap": parser.method_code,
        "comments": [],  # Optional: can be added
        "sinks": list(parser.method_calls.values())
    }

if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        file_path = os.sys.argv[1]
    else:
        file_path = "backend/src/bert/inference/attack_surface_detection.py"
    output = parse_python_file(file_path)
    print(json.dumps(output, indent=2))
