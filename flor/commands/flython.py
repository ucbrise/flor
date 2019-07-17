import os

import shutil
import importlib
from uuid import uuid4

from flor.versioner.versioner import Versioner
from flor.complete_capture.walker import Walker
from flor.logs_manipulator.open_log import OpenLog
from flor.constants import *

def exec_flython(args):

    if not os.path.exists(os.path.join(FLOR_DIR, '.conda_map')):
        print("Flor hasn't been installed.")
        print("From Python: You may run the function flor.install()")
        print("From CLI: You may run the pyflor_install script")
        import sys; sys.exit(0)

    # Get path and check
    full_path = os.path.abspath(args.path)
    assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    # Commit to repo before code transformation
    versioner = Versioner(full_path)
    versioner.save_commit_event("flor commit")

    # Transform code
    walker = Walker(os.path.dirname(full_path))
    walker.compile_tree(lib_code=False) # Transformed code in walker.targetpath

    # Model OpenLog Behavior
    ol = OpenLog(args.name, args.depth_limit)

    # Run code
    try:
        spec = importlib.util.spec_from_file_location("_{}".format(uuid4().hex), full_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        del module
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        print("Cleaning up...")

    # Model OpenLog Behavior TODO Add some fault tolerance
    ol.exit()

    # Restore original
    versioner.reset_hard()
