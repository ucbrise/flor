#!/usr/bin/env python3

import os
import warnings
from tqdm import tqdm
import pandas as pd
import tempfile
import shutil
import dill

from typing import Dict, Union, Optional, List

from . import global_state
from . import util
from jarvis.stateful import State
from jarvis.object_model.artifact import Artifact

def setNotebookName(name):
    global_state.nb_name = name

def getExperimentVersions(experimentName):
    versioningDir = State().versioningDirectory
    processed_out = []
    original_dir = os.getcwd()
    os.chdir(versioningDir + '/' + experimentName)
    raw_out = [x for x in util.runProc('git log').split('\n') if x]
    for line in raw_out:
        if len(line) >= 6 and line[0:6] == 'commit':
            processed_out.append(line.split(' ')[1])
    os.chdir(original_dir)
    return processed_out

def diffExperimentVersions(experimentName, v1, v2):
    original_dir = os.getcwd()
    os.chdir(State().versioningDirectory + '/' + experimentName)
    response = util.runProc('git diff ' + v1 + ' ' + v2 + ' ' + '0/' + experimentName + '.py')
    
    os.chdir(original_dir)

    response = response.split('\n')

    segmented_response = []
    file_level_dat = []

    for line in response:
        if len(line) >= 4 and line[0:4] == 'diff':
            if file_level_dat:
                segmented_response.append(file_level_dat)
            file_level_dat = []
        file_level_dat.append(line)

    def isCode(file_level_dat):
        intro = file_level_dat[0].split(' ')
        _, _, old, new = intro
        old = old.split('/')[-1]
        new = new.split('/')[-1]
        old_ext = old.split('.')[-1]
        new_ext = new.split('.')[-1]
        return old_ext == 'py' or new_ext == 'py'

    def getName(file_level_dat):
        intro = file_level_dat[0].split(' ')
        _, _, old, new = intro
        old = old.split('/')[-1]
        new = new.split('/')[-1]
        return old + ' --> ' + new

    def getBody(file_level_dat):
        active = False
        output = [getName(file_level_dat), ]
        for line in file_level_dat:
            if len(line) >= 2 and line[0:2] == '@@':
                active = True
            if active:
                output.append(line)
        return output

    filtered_segmented_response = list(filter(lambda x: isCode(x), segmented_response))

    name_body = {}

    for file_level_dat in filtered_segmented_response:
        if getName(file_level_dat) not in name_body:
            name_body[getName(file_level_dat)] = getBody(file_level_dat)

    res = list(name_body.values())

    color_res = []

    for file_level_dat in res:
        c_file_level_dat = []
        for line in file_level_dat:
            if len(line) >= 1 and line[0:1] == '-':
                c_line = "\033[1;31m" + line
                c_file_level_dat.append(c_line)
            elif len(line) >= 1 and line[0:1] == '+':
                c_line = "\033[1;32m" + line
                c_file_level_dat.append(c_line)
            else:
                c_file_level_dat.append("\033[0;30m" + line)
        color_res.append(c_file_level_dat)

    color_res = ['\n'.join(i) for i in color_res]

    for r in color_res:
        print(r)

def versionSummaries(experimentName):
    original_dir = os.getcwd()
    processed_out = []
    os.chdir(State().versioningDirectory + '/' + experimentName)
    for version in tqdm(getExperimentVersions(experimentName)):
        util.runProc('git checkout ' + version)
        df = util.loadArtifact(experimentName + '.pkl')
        processed_out.append((version, df))
    util.runProc('git checkout master')
    os.chdir(original_dir)

    for experiment_pair in processed_out:
        commithash, df = experiment_pair
        df.loc[:, '__commitHash__'] = [commithash for i in range(len(df))]

    processed_out = list(map(lambda x: x[1], processed_out))
    processed_out = pd.concat(processed_out).reset_index(drop=True)

    return processed_out

def checkoutArtifact(experimentName, trialNum, commitHash, fileName):
    original_dir = os.getcwd()
    os.chdir(State().versioningDirectory + '/' + experimentName)
    util.runProc('git checkout ' + commitHash)
    os.chdir(str(trialNum))
    warnings.filterwarnings("ignore")
    res = util.loadArtifact(fileName)
    warnings.filterwarnings("default")
    os.chdir('../')
    util.runProc('git checkout master')
    os.chdir(original_dir)
    return res

def run(experimentName : str, artifactLoc : str,
        squashMap : Dict[str, List[str]], commitHash : Optional[Union[List[str], str]] = None):

    with tempfile.TemporaryDirectory() as sharedtemp:

        resourceMap = {}

        xpdir = State().versioningDirectory + '/' + experimentName

        for currentPath in squashMap:
            currentPath2 = os.path.abspath(currentPath)
            filename = os.path.basename(currentPath2)
            shutil.copy(currentPath2, sharedtemp)
            resourceMap[currentPath2] = squashMap[currentPath]

        with util.chinto(xpdir):

            beforeTrialNum = len([x for x in os.listdir() if util.isNumber(x)])
            before = pd.DataFrame(columns=['version'] + [i for i in range(beforeTrialNum)])

            afterTrialNum = None
            after = None


            if commitHash:
                if type(commitHash) == str:
                    commitHashes = [commitHash,]
                else:
                    commitHashes = commitHash
            else:
                commitHashes = [x.split(' ')[1] for x in util.runProc('git log').split('\n')
                       if len(x) >= 6 and x[0:6] == 'commit']

            for version in tqdm(commitHashes):
                util.runProc('git checkout' + version)

                record = {'version': version}

                for i in range(beforeTrialNum):
                    record[i] = util.loadArtifact(str(i) + '/' + artifactLoc)

                before = before.append([record,]).loc[:, ['version'] + [i for i in range(beforeTrialNum)]]

                with tempfile.TemporaryDirectory() as tempdir:
                    shutil.copytree(xpdir, tempdir + '/' + experimentName)
                    with util.chinto(tempdir + '/' + experimentName + '/0' ):
                        with open('experiment_graph.pkl', 'rb') as f:
                            eg = dill.load(f)
                        for node in eg.d:
                            if type(node) == Artifact and node.loc == artifactLoc:
                                pullnode = node
                                break
                        for absfilepath in resourceMap:
                            for relativefilepath in resourceMap[absfilepath]:
                                shutil.copy(absfilepath, relativefilepath)
                        try:
                            pullnode.pull()
                        except Exception as e:
                            print("Exception in version: {}\nError message: {}".format(version, e))

                # Now back in xpdir
                util.runProc('git branch garbage')
                util.runProc('git checkout master')
                util.runProc('git merge garbage --no-edit')
                util.runProc('git branch -d garbage')

                raw = util.runProc('git log').split('\n')
                if len(raw) > 1 and 'Merge' in raw[1]:
                    #get second commit
                    newCommitHashes = [x.split(' ')[1] for x in raw if len(x) >= 6 and x[0:6] == 'commit']
                    newVersion = newCommitHashes[1]
                    del newCommitHashes
                else:
                    #Get first commit
                    newVersion = raw[0].split(' ')[1]

                if after is None:
                    afterTrialNum = len([x for x in os.listdir() if util.isNumber(x)])
                    after  = pd.DataFrame(columns=['version', 'newVersion'] + ['n_' + str(i) for i in range(afterTrialNum)])

                record = {'version': version, 'newVersion': newVersion}

                for i in range(afterTrialNum):
                    record['n_' + str(i)] = util.loadArtifact(str(i) + '/' + artifactLoc)

                after = after.append([record,]).loc[:, ['version', 'newVersion'] + ['n_' + str(i) for i in range(afterTrialNum)]]

    return before.merge(after, left_on='version', right_on='version', how='inner')




