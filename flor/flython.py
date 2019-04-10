import argparse
import os
import shutil

from flor.versioner.versioner import Versioner
from flor.dynamic_capture.walker import Walker
from flor.face_user.open_log import OpenLog

parser = argparse.ArgumentParser()
parser.add_argument("path", help="The path to the model training pipeline to execute")
parser.add_argument("name", help="The name of the experiment to run")
parser.add_argument("-d", "--depth_limit", type=int, help="Depth limit the logging")


args = parser.parse_args()

if __name__ == '__main__':
    # Get path and check
    full_path = os.path.abspath(args.path)
    assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    # Commit to repo before code transformation
    versioner = Versioner(full_path)
    versioner.save_commit_event("flor commit")

    # Transform code
    walker = Walker(os.path.dirname(full_path))
    walker.compile_tree(lib_code=False) # Transformed code in walker.targetpath

    # Overwrite current directory
    current_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(full_path)))
    shutil.rmtree(os.path.dirname(full_path))
    shutil.copytree(walker.targetpath, os.path.dirname(full_path))
    os.chdir(current_dir)

    # Model OpenLog Behavior
    ol = OpenLog(args.name, args.depth_limit)

    # Run code
    exec(open(full_path).read())

    # Model OpenLog Behavior TODO Add some fault tolerance
    ol.exit()

    # Restore original
    versioner.reset_hard()
