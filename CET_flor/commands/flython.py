import warnings
import importlib
from uuid import uuid4

from flor.versioner.versioner import Versioner
from flor.complete_capture.walker import Walker
from flor.logs_manipulator.open_log import OpenLog
from flor.constants import *

import sys, os
# Adding to global namespace
import flor
from flor import Flog

def exec_flython(args):
    if not os.path.exists(os.path.join(FLOR_DIR, '.conda_map')):
        print("Flor hasn't been installed.")
        print("From Python: You may run the function flor.install()")
        print("From CLI: You may run the pyflor_install script")
        sys.exit(0)

    # Get path and check
    full_path = os.path.abspath(args.path)
    assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    # Commit to repo before code transformation
    versioner = Versioner(full_path)
    versioner.save_commit_event("flor commit on exec")

    # Model OpenLog Behavior
    ol = OpenLog(args.name)

    try:
        with open(full_path, 'r') as f:
            full_text = f.read()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sys.path.insert(0, os.path.dirname(full_path))
            sys.argv = sys.argv[3:-1]
            g = globals()
            old_name = g.get('__name__', None)
            l = {'__name__': '__main__'}
            g.update(l)
            exec(full_text, g, g)
            if old_name is None:
                del g['__name__']
            else:
                g['__name__'] = old_name
            sys.path.pop(0)
    except:
        import traceback
        e = sys.exc_info()[0]
        traceback.print_exc()
        print(e)
        print("Cleaning up...")
    finally:
        # Model OpenLog Behavior TODO Add some fault tolerance
        ol.exit()

    # Now re-exec
    print("Execution finished... re-executing...")

def re_exec_flython(args):
    if not os.path.exists(os.path.join(FLOR_DIR, '.conda_map')):
        print("Flor hasn't been installed.")
        print("From Python: You may run the function flor.install()")
        print("From CLI: You may run the pyflor_install script")
        sys.exit(0)

    # Get path and check
    full_path = os.path.abspath(args.path)
    assert os.path.splitext(os.path.basename(full_path))[1] == '.py'

    # Commit to repo before code transformation
    versioner = Versioner(full_path)
    versioner.save_commit_event("flor commit on re-exec")

    # Transform code
    walker = Walker(os.path.dirname(full_path))
    walker.compile_tree(lib_code=False) # Transformed code in walker.targetpath

    # Model OpenLog Behavior
    ol = OpenLog(args.name, args.depth_limit)
    ol.set_re_execution_flag()

    # Run code
    try:
        with open(full_path, 'r') as f:
            full_text = f.read()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sys.path.insert(0, os.path.dirname(full_path))
            sys.argv = sys.argv[3:-1]
            g = globals()
            old_name = g.get('__name__', None)
            l = {'__name__': '__main__'}
            g.update(l)
            exec(full_text, g, g)
            if old_name is None:
                del g['__name__']
            else:
                g['__name__'] = old_name
            sys.path.pop(0)
    except:
        import traceback
        e = sys.exc_info()[0]
        traceback.print_exc()
        print(e)
        print("Cleaning up...")
    finally:
        # Model OpenLog Behavior TODO Add some fault tolerance
        ol.exit()

        # Restore original
        versioner.reset_hard()
