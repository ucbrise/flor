#!/usr/bin/env python3

import subprocess
import os
import uuid

from typing import List

from . import globals

class chinto(object):
    def __init__(self, target):
        self.original_dir = os.getcwd()
        self.target = target

    def __enter__(self):
        os.chdir(self.target)

    def __exit__(self, type, value, traceback):
        os.chdir(self.original_dir)

class chkinto(object):
    def __init__(self, commit):
        self.target = commit
        self.currentBranch = __get_current_branch__()

    def __enter__(self):
        subprocess.run(['git', 'checkout', self.target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __exit__(self, type, value, traceback):
        subprocess.run(['git', 'checkout', self.currentBranch], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def gitlog(sourceKey, typ):
    typ = typ.lower()
    ld = []
    with chinto(os.path.join(globals.GRIT_D, typ, sourceKey)):
        rawgitlog = __readProc__(['git', 'log', '--all'])
        d = {}
        for line in rawgitlog:
            if 'commit' in line[0:6]:
                d['commit'] = line.split(' ')[1]
            elif 'Author' in line[0:6]:
                d['Author'] = ' '.join(line.split()[1:])
            elif 'Date' in line[0:4]:
                d['Date'] = ' '.join(line.split()[1:])
            elif 'id:' in line and 'class:' in line:
                line = line.split()
                d['id'] = int(line[1].split(',')[0])
                d['class'] = line[3]
                ld.append(d)
                d = {}
    return ld

def gitdag(sourceKey, typ):
    typ = typ.lower()
    ld = []
    with chinto(os.path.join(globals.GRIT_D, typ, sourceKey)):
        rawgitlog = __readProc__(['git', 'rev-list', '--children', '--all', '--pretty=format:%s'])
        d = {}
        for line in rawgitlog:
            fields = line.split()
            if 'commit' in line[0:6]:
                d['commit'] = fields[1]
                d['children'] = fields[2:]
            elif 'id:' in line and 'class:' in line:
                d['id'] = str(fields[1].split(',')[0])
                d['class'] = fields[3]
                ld.append(d)
                d = {}

    # Remove the class: Node dummy that is first commit
    ld = ld[0:-1]

    # Partition list into Merge commit and NodeVersion commit:
    nvs = []
    ms = []
    for d in ld:
        if d['class'] == 'Merge':
            ms.append(d)
        elif 'Version' in d['class']:
            nvs.append(d)
        else:
            raise ValueError("Unexepected class: {}".format(d['class']))

    # Build a map
    # Commit -> (Desired Id)
    d2 = {}
    for nv in nvs:
        d2[nv['commit']] = nv['id']
    for m in ms:
        assert len(m['children']) == 1
        d2[m['commit']] = d2[m['children'][0]]

    # Prepare final output
    fd = {}
    for nv in nvs:
        fd[d2[nv['commit']]] = set(map(lambda x: d2[x], nv['children']))

    return fd


def get_ver_commits(sourceKey, typ):

    ld = gitlog(sourceKey, typ)
    return map(lambda x: (x['commit'], x['id']),
                    filter(lambda x: 'Version' in x['class'],
                           ld))

def get_commits(sourceKey, typ):
    ld = gitlog(sourceKey, typ)
    return list(map(lambda x: (x['commit'], x['id']), ld))

def get_branch_commits(sourceKey, typ):
    # Warning: returns iterator not list (not subscriptable)
    with chinto(os.path.join(globals.GRIT_D, typ, sourceKey)):
        def clean(s):
            s = s.strip()
            if '* ' in s:
                s = s[2:]
            return s
        branches = [i for i in map(clean, __readProc__(['git', 'branch'])) if i]
        commits = [i for i in __readProc__(['git', 'rev-parse', '--branches']) if i]

    return zip(branches, commits)

def id_to_commit(id, sourcekey, typ):
    for commit, id2 in get_ver_commits(sourcekey, typ):
        if int(id2) == int(id):
            return commit

def __get_current_branch__():
    def split(s):
        s = s.strip()
        return s.split()
    return [i for i in map(split, __readProc__(['git', 'branch'])) if len(i) == 2][0][1]

def new_branch_name(sourceKey, typ):
    with chinto(os.path.join(globals.GRIT_D, typ, sourceKey)):
        rawgitlog = __readProc__(['git', 'branch'])
        new_name = uuid.uuid4().hex
        while new_name in rawgitlog:
            new_name = uuid.uuid4().hex

    return new_name

def __runProc__(commands: List):
    subprocess.run(commands, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

def __readProc__(commands: List):
    p1 = subprocess.run(commands, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rawgitlog = str(p1.stdout, 'UTF-8').split('\n')
    return rawgitlog

def runThere(commands: List, sourceKey, typ):
    with chinto(os.path.join(globals.GRIT_D, typ, sourceKey)):
        __runProc__(commands)

def readThere(commands: List, sourceKey, typ):
    with chinto(os.path.join(globals.GRIT_D, typ, sourceKey)):
        out = __readProc__(commands)
    return out