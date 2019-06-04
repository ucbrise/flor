import os
import ast

from flor.complete_capture.transformer import ClientTransformer
from flor.state_machine_generator import Visitor

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

        # Scan the log
        visitor.scanner.scan_log()
        df = visitor.scanner.to_df()
        #TODO: we will want to join the tables across file highlights
            # This corresponds to a function in one file calling out to a function in another file
        #TODO : To join the tables in full_paths need to do DataFlow analysis
        target, _= os.path.splitext(os.path.basename(full_path))
        df.to_csv(target + '.csv')
