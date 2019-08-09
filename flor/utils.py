import shutil
from flor.constants import *
import os


def cond_mkdir(path):
    """
    Mkdir if not exists
    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def refresh_tree(path):
    """
    When finished, brand new directory root at path
        Whether or not it used to exist and was empty
    :param path:
    :return:
    """
    cond_rmdir(path)
    os.mkdir(path)


def cond_rmdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def check_flor_install():
    if not os.path.exists(os.path.join(FLOR_DIR, '.conda_map')):
        print("Flor hasn't been installed.")
        print("From Python: You may run the function flor.install()")
        print("From CLI: You may run the pyflor_install script")
        import sys
        sys.exit(0)

def write_debug_msg(msg):
    assert isinstance(msg, str)
    with open(os.path.join(FLOR_DIR, 'debug_msg.txt'), 'a') as f:
        f.write(msg + '\n')

def write_failure_msg(msg):
    assert isinstance(msg, str)
    with open(os.path.join(FLOR_DIR, 'failures.txt'), 'a') as f:
        f.write(msg + '\n')