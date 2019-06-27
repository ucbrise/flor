import os
from shutil import copyfile
from flor.complete_capture.walker import Walker

examples_dir = 'tests/examples/'

src_file_path = examples_dir + 'iris_raw.py'

temp_file_path = examples_dir + 'temp/iris_raw.py'

src_file_transformed_path = examples_dir + 'iris_raw_tf.py'


def test_exec_flython():
    pass


# Test walker
def test_transform():
    temp_folder_path = os.path.dirname(temp_file_path)
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)

    copyfile(src_file_path, temp_file_path)

    full_path = os.path.abspath(temp_file_path)
    assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    walker = Walker(os.path.dirname(full_path))
    walker.compile_tree(lib_code=False)

    mismatch_lines = []

    with open(temp_file_path) as f1, open(src_file_transformed_path) as f2:
        for lineno, (line1, line2) in enumerate(zip(f1, f2), 1):
            if line1 != line2:
                mismatch_lines.append((lineno))

    num_mismatched_lines = len(mismatch_lines)
    assert num_mismatched_lines <= 2
    if num_mismatched_lines == 1:
        assert mismatch_lines == [5]
    elif num_mismatched_lines == 2:
        assert mismatch_lines == [5, 6]

    os.remove(temp_file_path)
