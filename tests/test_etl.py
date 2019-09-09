import os
import sys
from argparse import Namespace
from flor.__main__ import main
from flor.commands.flan import exec_flan
from flor.commands.cp import exec_cp
from flor.commands.flan import _get_src_filename

test_dir = 'tests/'
examples_dir = os.path.join(test_dir, 'examples/')

def test__get_src_filename():
    src_file_path = os.path.join(examples_dir, 'iris_raw.py')
    dst_file_path = os.path.join(examples_dir, 'iris_raw_clone.py')

    exec_cp(Namespace(src=src_file_path, dst=dst_file_path))

    assert _get_src_filename(dst_file_path).split('flor/')[1] == 'tests/examples/iris_raw.py'

    # delete cloned dst_file
    os.remove(dst_file_path)


def test_etl():
    exp_name = 'etl_test_exp_name'
    # run flor python
    sys.argv = ['', 'python', examples_dir + 'iris_etl.py', exp_name]
    main()
    src_file_path = os.path.join(examples_dir, 'iris_etl.py')
    h_file_path = os.path.join(examples_dir, 'iris_etl_h.py')
    # replace first line with data header
    with open(h_file_path) as f:
        lines = f.readlines()
    lines[0] = '#' + os.path.abspath(src_file_path)
    with open(h_file_path, 'w') as f:
        f.writelines(lines)
    print(open(h_file_path).read())
    # run flor etl
    exec_flan(Namespace(name=exp_name, annotated_file=[os.path.abspath(h_file_path)]))
    assert len(open(exp_name + '.csv', 'r').read().strip()) > 2, 'CSV file generated must be non-empty'
    