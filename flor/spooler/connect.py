import os
import re
import subprocess
from .. import stateful as flags
from flor.spooler.utils import natural_key, convert_to_int

flor_dir = os.path.expanduser(os.path.join('~', '.flor'))

def send_to_S3():
    path = os.path.join(flor_dir, 'send_to_S3.sh')
    index = flags.LOG_PATH.absolute.rindex('/')
    index_path = flags.LOG_PATH.absolute[:index]
    index_name = flags.LOG_PATH.absolute[index+1:]

    file_names = os.listdir(index_path)
    file_names.sort(key=natural_key)

    index_file = open(flags.LOG_PATH.absolute, 'a+')
    for fn in file_names:
        if fn[:-5].find('.') != -1 and fn.find(index_name[:-5]) != -1:
            fp = open(os.path.join(index_path, fn), 'r')
            index_file.write(fp.read())
            fp.close()
            os.remove(os.path.join(index_path, fn))
    index_file.close()

    subprocess.run(['bash', path, str(flags.NAME), str(index_name)], check=True)

def receive_index_from_S3():
    path = os.path.join(flor_dir, 'receive_index_from_S3.sh')
    index = flags.MEMO_PATH.absolute.rindex('/')
    index_name = flags.MEMO_PATH.absolute[index+1:]
    subprocess.run(['bash', path, str(flags.NAME), str(index_name)], check = True)

def receive_from_S3():
    receive_index_from_S3()
    path = os.path.join(flor_dir, 'receive_from_S3.sh')
    with open(flags.MEMO_PATH.absolute, 'r') as fp:
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
