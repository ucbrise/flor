from flor.complete_capture.walker import Walker
from filecmp import dircmp
from flor.constants import *

import shutil
import errno


def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def test_in_place_transformation():
    with open(os.path.join(FLOR_DIR, '.conda_map'), 'r') as f:
        src_root, dst_root = f.read().strip().split(',')

    sklearn_dir = os.path.join(dst_root, 'lib/python3.6/site-packages/sklearn')

    # Create a copy of the transformed sklearn dir to transform again
    sklearn_transformed_dir = sklearn_dir + '-transformed'

    copy(sklearn_dir, sklearn_transformed_dir)

    walker = Walker(sklearn_transformed_dir)
    walker.compile_tree()

    num_diff_files = len(dircmp(sklearn_transformed_dir, sklearn_dir).diff_files)

    shutil.rmtree(sklearn_transformed_dir)

    assert num_diff_files == 0, 'compile_tree() should not transform already-transformed libraries.'


if __name__ == '__main__':
    test_in_place_transformation()
