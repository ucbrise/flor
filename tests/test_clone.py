import os
from argparse import Namespace
from flor.commands.cp import exec_cp
import filecmp

examples_dir = 'tests/examples/'


def test_exec_cp():
    src_file_path = examples_dir + 'iris_raw.py'
    dst_file_path = examples_dir + 'iris_raw_h.py'

    exec_cp(Namespace(src=src_file_path, dst=dst_file_path))

    # assert filecmp.cmp(dst_file_path, test_dir + 'iris_raw_h_exp.py')

    lineSet = set()
    with open(src_file_path) as f:
        for line in f:
            lineSet.add(line)

    with open(dst_file_path) as f:
        # Ignore first line with flor-specific metadata
        f.readline()

        for line in f:
            if line in lineSet:
                lineSet.remove(line)

    # dst_file should be same as src_file except first line
    assert len(lineSet) == 0, "exec_cp cloned file was different from src file on more than line 1"

    # delete cloned dst_file
    os.remove(dst_file_path)
