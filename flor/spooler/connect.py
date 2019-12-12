import subprocess
from .. import stateful as flags
import re
import os

flor_dir = os.path.expanduser(os.path.join('~', '.flor'))

def send_to_S3():
    path = os.path.join(flor_dir, 'send_to_S3.sh')
    subprocess.run(['bash', path, str(flags.NAME)], check=True)

def receive_from_S3():
    path = os.path.join(flor_dir, 'receive_from_S3.sh')
    with open(flags.LOG_PATH.absolute, 'r') as fp:
        line = fp.readline()
        while line:
            re_pkl = re.search(r'\.pkl', line)
            if re_pkl:
                words = line.split()
                for word in words:
                    if re.match('.+\.pkl', word):
                        word = word.strip(',')
                        word = word.strip('\"')
                        index = word.rindex('/')
                        pkl_name = word[index+1:]
                        subprocess.run(['bash', path, os.path.expanduser(str(word)), str(flags.NAME), str(pkl_name)], check=True)
            line = fp.readline()
