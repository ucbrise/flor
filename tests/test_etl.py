import os
from argparse import Namespace
from flor.commands.flan import exec_flan
from flor.commands.cp import exec_cp
from flor.commands.flan import _get_src_filename

test_dir = 'tests/'
examples_dir = test_dir + 'examples/'

def test__get_src_filename():
    src_file_path = examples_dir + 'iris_raw.py'
    dst_file_path = examples_dir + 'iris_raw_clone.py'

    exec_cp(Namespace(src=src_file_path, dst=dst_file_path))

    assert _get_src_filename(dst_file_path).split('flor/')[1] == 'tests/examples/iris_raw.py'

    # delete cloned dst_file
    os.remove(dst_file_path)