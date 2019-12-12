from subprocess import Popen
from .. import stateful as flags
import re
import os

flor_dir = os.path.expanduser(os.path.join('~', '.flor'))

def send_to_S3():
    path = os.path.join(flor_dir, 'send_to_S3.sh')
    Process = Popen('bash %s %s' % (path, str(flags.NAME),), shell=True)

def receive_from_S3():
    try:
        path = os.path.join(flor_dir, 'receive_from_S3.sh')
        fp = open(flags.LOG_PATH.absolute, 'r')
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
                        Process = Popen('bash %s %s %s %s' % (path, str(word),str(flags.NAME),str(pkl_name),), shell=True)
            line = fp.readline()
    finally:
        fp.close()
