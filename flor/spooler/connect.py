from subprocess import Popen
from .. import stateful as flags
import re

def send_to_S3():
    Process = Popen('../flor/flor/spooler/send_to_S3.sh %s' % (str(flags.NAME),), shell=True)

def receive_from_S3():
    try:
        #fp = open('../.flor/resnet18-s3/20191121-202158.json', 'r')
        fp = open(flags.LOG_PATH, 'r')
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
                        Process = Popen('../flor/flor/spooler/receive_from_S3.sh %s %s %s' % (str(word),str(flags.NAME),str(pkl_name),), shell=True)
            line = fp.readline()
    finally:
        fp.close()

