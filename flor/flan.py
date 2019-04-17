import os
import ast, astor

from flor.dynamic_capture.transformer import ClientTransformer

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
    print("Log found at {}".format(log_path))

    for full_path in full_paths:
        # Transform code
        exec_path = _get_src_filename(full_path)
        transformer = ClientTransformer(exec_path)
        with open(full_path, 'r') as f:
            astree = ast.parse(f.read())
        new_astree = transformer.visit(astree)

        dir_name = os.path.dirname(full_path)
        base_name = os.path.basename(full_path)
        *path, ext = base_name.split('.')
        path.append('annotated')
        path.append(ext)
        new_base_name = '.'.join(path)

        target_full_path = os.path.join(dir_name, new_base_name)
        with open(target_full_path, 'w') as f:
            f.write("#" + exec_path + "\n")
            f.write(astor.to_source(new_astree))
