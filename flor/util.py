#!/usr/bin/env python3

import subprocess
import pickle
import hashlib
import os

from typing import  List

def isLoc(loc):
    try:
        ext = loc.split('.')[1]
        return True
    except:
        return False


def isFlorClass(obj):
    return type(obj).__name__ == "Artifact" or type(obj).__name__ == "Action" or type(obj).__name__ == "Literal"


def isLiteral(obj):
    return type(obj).__name__ == "Literal"


def isPickle(loc):
    try:
        return loc.split('.')[1] == 'pkl'
    except:
        return False


def isCsv(loc):
    try:
        return loc.split('.')[1] == 'csv'
    except:
        return False


def isIpynb(loc):
    return loc.split('.')[1] == 'ipynb'


def isIterable(obj):
    return type(obj) == list or type(obj) == tuple


def runProc(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    try:
        return str(output, 'UTF-8')
    except:
        return output


def pickleTo(obj, loc):
    with open(loc, 'wb') as f:
        pickle.dump(obj, f)


def unpickle(loc):
    with open(loc, 'rb') as f:
        x = pickle.load(f)
    return x


def isOrphan(obj):
    return obj.parent is None


def isNumber(s):
    if type(s) == int or type(s) == float:
        return True
    try:
        float(s)
        return True
    except:
        return False


def loadArtifact(loc):
    if isPickle(loc):
        x = unpickle(loc)
    elif isLoc(loc):
        with open(loc, 'r') as f:
            x = [i.strip() for i in f.readlines() if i.strip()]
        if len(x) == 1:
            x = x[0]
    else:
        raise NotImplementedError("Don't know how to load that file.")
    return x


def master_pop(literals):
    if not literals:
        return True
    subtreeMaxed = master_pop(literals[0:-1])
    if subtreeMaxed:
        popSuccess = literals[-1].__pop__()
        if not popSuccess:
            return True
        [literal.__reset__() for literal in literals[0:-1]]
    return False


def activate(pseudoArtifact):
    from flor.object_model.literal import Literal
    if type(pseudoArtifact) == Literal:
        pseudoArtifact.__enable__()
    elif not isOrphan(pseudoArtifact) and pseudoArtifact.parent.in_artifacts:
        for in_art in pseudoArtifact.parent.in_artifacts:
            activate(in_art)


def md5(fname):
    # Credit: https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def plating(in_artifacts):
    multiplier = []
    for _in in in_artifacts:
        if isLiteral(_in) and _in.__oneByOne__:
            value = len(_in.v)
            if value > 1:
                multiplier.append(str(len(_in.v)))
    plate_label = 'x'.join(multiplier)
    if plate_label:
        return plate_label
    return None


class chinto(object):
    def __init__(self, target):
        self.original_dir = os.getcwd()
        self.target = target

    def __enter__(self):
        os.chdir(self.target)

    def __exit__(self, type, value, traceback):
        os.chdir(self.original_dir)

def git_log():
    ld = []
    rawgitlog = __readProc__(['git', 'log', '--all']).split('\n')
    d = {}
    for line in rawgitlog:
        if 'commit' in line[0:6]:
            d['commit'] = line.split(' ')[1]
        elif 'Author' in line[0:6]:
            d['Author'] = ' '.join(line.split()[1:])
        elif 'Date' in line[0:4]:
            d['Date'] = ' '.join(line.split()[1:])
        elif 'msg:' in line.strip()[0:len('msg:')]:
            line = line.strip()
            d['message'] = ':'.join(line.split(':')[1:])
        if d:
            ld.append(d)
        d = {}
    return ld

def __runProc__(commands: List):
    subprocess.run(commands, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

def __readProc__(commands: List):
    p1 = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rawgitlog = str(p1.stdout, 'UTF-8')
    return rawgitlog