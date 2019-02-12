import json
import logging
import os
import sys

from flor.controller.versioner import Versioner
from flor.controller.parser import injected
import flor.global_state as global_state

logger = logging.getLogger(__name__)

class Context():
    """
    Defines a `context manager <http://book.pythontips.com/en/latest/context_managers.html>`_.

    The top-level flor decorated function in some experiment.
    Every top-level flor-decorated function must be called from within a Flor Context.

    Usage:

    https://github.com/ucbrise/flor/tree/master/examples

    :param xp_name: The name of the experiment, unique in the scope of the Flor user
    """


    def __init__(self, xp_name: str):
        assert not global_state.interactive, "flor.Context not defined for interactive environments (IPython or Jupyter)"
        self.xp_name = xp_name
        self.versioner = Versioner()

        if not self.versioner.get_ancestor_repo_path(os.getcwd()):
            logger.warning("The current working directory {} is not a git repository.\n".format(os.getcwd()) +
                           "We recommend that you quit this program. Initialize a git repository, " +
                           "and add a `.gitignore` file")
            while True:
                quit_flag = input("Would you like to quit? [y/N] ").strip()
                if quit_flag == 'N':
                    break
                elif quit_flag.lower() == 'y':
                    logger.info("Quitting...")
                    sys.exit(0)
                else:
                    logger.info('Invalid entry: {}'.format(quit_flag))

        self.log_file_name = os.path.abspath('{}_log.json'.format(self.xp_name))
        injected.file = open(self.log_file_name, 'w')


    def __enter__(self):
        assert not global_state.interactive, "flor.Context not defined for interactive environments (IPython or Jupyter)"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert not global_state.interactive, "flor.Context not defined for interactive environments (IPython or Jupyter)"
        injected.file.close()
        self.versioner.save_commit_event("Experiment Name :: {}\n\n".format(self.xp_name), self.log_file_name)