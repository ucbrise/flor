import os
import ast

from flor.complete_capture.transformer import ClientTransformer
from flor.state_machine_generator import Visitor

class DebuggingObj:
    def __init__(self, name, abspath_root, annotated_files: 'List'):
        assert isinstance(annotated_files, list)
        annotated_files = [os.path.join(abspath_root, a_f) for a_f in annotated_files]
        self.name = name
        self.annotated_file = annotated_files

    def debug(self):
        exec_flan(self)

def _get_src_filename(full_path):
    with open(full_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split('#')
    #TODO: This is where we check that the file we are reading is valid output of Flor highlight
    assert len(parts) > 1, "Invalid annotation file. Did you call `flor cp source.py target.py` first?\n" \
                           "The annotation file is `target.py`."
    return parts[1]

def exec_flan(args):
    # Get path and check
    full_paths = [os.path.abspath(path) for path in args.annotated_file]
    for full_path in full_paths:
        assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    # Get Log Name
    log_path = os.path.join(os.path.expanduser('~'), '.flor', args.name, 'log.json')
    assert os.path.exists(log_path)

    visitors = []
    for full_path in full_paths:
        # Transform code TODO: what if library code is annotated? ClientTransformer will mismatch
        #Solution: flor highlighter can provide info about where code comes from
        exec_path = _get_src_filename(full_path)
        transformer = ClientTransformer(exec_path)
        with open(full_path, 'r') as f:
            astree = ast.parse(f.read())
        new_astree = transformer.visit(astree)

        # Generate State machines
        visitor = Visitor(exec_path, log_path)
        visitor.visit(new_astree)
        visitors.append(visitor)

    consolidated_scanner = visitors.pop(0).scanner
    for visitor in visitors:
        consolidated_scanner.state_machines.extend(visitor.scanner.state_machines)

    # Scan the log
    consolidated_scanner.scan_log()
    df = consolidated_scanner.to_df()
    target = args.name
    df.to_csv(target + '.csv')

if __name__ == '__main__':
    name = "bgfx"
    DebuggingObj(name, 
        "/Users/rogarcia/sandbox/iris_loop",
        ['iris_h.py', 'sklearn_svm_classes_h.py']).debug()
    if os.path.exists(name + '.csv'):
        os.remove(name+'.csv')
