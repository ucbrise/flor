import ast, astor

class Scanner:

    def __init__(self, annotated_file_path):
        self.annotated_file_path = annotated_file_path
        self.lines = []

    def scan_file(self):
        with open(self.annotated_file_path, 'r') as f:
            astree = ast.parse(f.read())


