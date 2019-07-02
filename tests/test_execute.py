import os
import shutil
from shutil import copyfile
import contextlib
import tempfile

from flor.complete_capture.walker import Walker

from .settings import _DIGITS_RAW_FILE, _EXAMPLES_DIR


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)

    with cd(dirpath, cleanup):
        yield dirpath


temp_dir = os.path.join(_EXAMPLES_DIR, 'temp/')

src_file_path = os.path.join(_EXAMPLES_DIR, 'iris_raw.py')

temp_file_path = os.path.join(temp_dir, 'iris_raw.py')

src_file_transformed_path = os.path.join(_EXAMPLES_DIR, 'iris_raw_tf.py')


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
                mismatch_lines.append(lineno)

    with open(src_file_path) as src_f, open(temp_file_path) as dst_f:
        src_lines = src_f.read().replace(' ', '').splitlines()
        dst_lines = dst_f.read().replace(' ', '')

        # Lines in source file should be in transformed file as well
        assert all([l in dst_lines for l in src_lines])

    num_mismatched_lines = len(mismatch_lines)

    os.remove(temp_file_path)


def test_transform_digits_raw():
    full_path = os.path.abspath(_DIGITS_RAW_FILE)
    assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    walker = Walker(full_path)
    walker.compile_tree(lib_code=False)
