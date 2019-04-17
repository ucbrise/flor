import os
import ast

from flor.dynamic_capture.transformer import ClientTransformer
from flor.state_machiner.visitor import Visitor

def _get_src_filename(full_path):
    with open(full_path, 'r') as f:
        line = f.readline().strip()
    return line.split('#')[1]

def exec_flan(args):
    # Get path and check
    full_paths = [os.path.abspath(path) for path in args.annotated_file]
    for full_path in full_paths:
        assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    # Get Log Name
    log_path = os.path.join(os.path.expanduser('~'), '.flor', args.name, 'log.json')
    assert os.path.exists(log_path)

    for full_path in full_paths:
        # Transform code
        exec_path = _get_src_filename(full_path)
        transformer = ClientTransformer(exec_path)
        with open(full_path, 'r') as f:
            astree = ast.parse(f.read())
        new_astree = transformer.visit(astree)

        # Generate State machines
        visitor = Visitor(exec_path, log_path)
        visitor.visit(new_astree)

        # Scan the log
        visitor.scanner.scan_log()
        df = visitor.scanner.to_df()
        #TODO: we will want to join the tables across file highlights
            # This corresponds to a function in one file calling out to a function in another file
        #TODO : To join the tables in full_paths need to do DataFlow analysis
        target, _= os.path.splitext(os.path.basename(full_path))
        df.to_csv(target + '.csv')