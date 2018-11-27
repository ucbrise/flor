#!/usr/bin/env python3

import subprocess
import hashlib
import os

from typing import  List

import json
import os.path
import re
import ipykernel
import requests


### GET JUPYTER NOTEBOOK NAME ###
# Alternative that works for both Python 2 and 3:
from requests.compat import urljoin

try:  # Python 3 (see Edit2 below for why this may not work in Python 2)
    from notebook.notebookapp import list_running_servers
except ImportError:  # Python 2
    import warnings
    from IPython.utils.shimmodule import ShimWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ShimWarning)
        from IPython.html.notebookapp import list_running_servers


def get_notebook_name():
    """
    https://github.com/jupyter/notebook/issues/1000#issuecomment-359875246
    Return the full path of the jupyter notebook.
    """
    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                params={'token': ss.get('token', '')})
        for nn in json.loads(response.text):
            if nn['kernel']['id'] == kernel_id:
                relative_path = nn['notebook']['path']
                return os.path.join(ss['notebook_dir'], relative_path)

### GET JUPYTER NOTEBOOK NAME ###

def runProc(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    try:
        return str(output, 'UTF-8')
    except:
        return output

def isNumber(s):
    if type(s) == int or type(s) == float:
        return True
    try:
        float(s)
        return True
    except:
        return False

def md5(fname):
    # Credit: https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def __runProc__(commands: List):
    subprocess.run(commands, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

def __readProc__(commands: List):
    p1 = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rawgitlog = str(p1.stdout, 'UTF-8')
    return rawgitlog