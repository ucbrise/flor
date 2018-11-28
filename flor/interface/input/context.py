import json
import logging
import os
import sys

from flor.controller.parser.injected import structured_log
from flor.controller.versioner import Versioner

logger = logging.getLogger(__name__)

class Context():
    def __init__(self, xp_name):
        self.xp_name = xp_name
        self.versioner = Versioner()

        if not self.versioner.get_ancestor_repo_path(os.getcwd()):
            logger.warning("The current working directory {} is not a git repository.\n" +
                           "We recommend that you quit this program. Initialize a git repository, " +
                           "and add a `.gitignore` file".format(os.getcwd()))
            while True:
                quit_flag = input("Would you like to quit? [y/N]").strip()
                if quit_flag == 'N':
                    break
                elif quit_flag.lower() == 'y':
                    logger.info("Quitting...")
                    sys.exit(0)
                else:
                    logger.info('Invalid entry: {}'.format(quit_flag))


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        log_file_name = os.path.abspath('{}_log.json'.format(self.xp_name))
        with open(log_file_name, 'w') as f:
            json.dump(structured_log.log_tree, f)
        self.versioner.save_commit_event("flor commit", log_file_name)