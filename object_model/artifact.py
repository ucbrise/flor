#!/usr/bin/env python3

import os
import git
import json
import itertools
import datetime
import pandas as pd
import sys

from graphviz import Digraph
from shutil import copyfile
from shutil import rmtree
from shutil import copytree
from shutil import move

from .. import util
from .. import above_ground as ag
from .recursivesizeof import total_size
import time

import ray

class Artifact:

    def __init__(self, loc, parent, manifest, xp_state):
        self.loc = loc
        self.parent = parent

        if self.parent:
            self.parent.out_artifacts.append(self)

        self.xp_state = xp_state

    def __commit__(self):

        gc = self.xp_state.gc #ground client
        dir_name = self.xp_state.versioningDirectory
        loclist = self.loclist
        scriptNames = self.scriptNames
        tag = {
            'Artifacts': [i for i in loclist], 
            'Actions': [i for i in scriptNames] 
        }

        for literal in self.xp_state.literals:
            if literal.name:
                try:
                    value = str(util.unpickle(literal.loc))
                    if len(value) <= 250:
                        tag[literal.name] = value 
                except:
                    pass

        if not os.path.exists(dir_name):
            nodeid = gc.createNode('Run')
            gc.createNodeVersion(nodeid, tag) 

            os.makedirs(dir_name) 
            os.makedirs(dir_name + '/1') 
            for loc in loclist:
                copyfile(loc, dir_name + "/1/" + loc)
            for script in scriptNames:
                copyfile(script, dir_name + "/1/" + script)
            os.chdir(dir_name + '/1')

            gc.commit() 
            os.chdir('../') 

            repo = git.Repo.init(os.getcwd()) 
            repo.index.add(['1',])

            repo.index.commit("initial commit")
            tree = repo.tree()
            with open('.jarvis', 'w') as f:
                for obj in tree:
                    commithash = util.runProc("git log " + obj.path).replace('\n', ' ').split()[1]
                    if obj.path != '.jarvis':
                        f.write(obj.path + " " + commithash + "\n")
            repo.index.add(['.jarvis'])
            repo.index.commit('.jarvis commit')
            os.chdir('../') 
        else:

            listdir = [x for x in filter(util.isNumber, os.listdir(dir_name))] 

            
            nthDir =  str(len(listdir) + 1)
            os.makedirs(dir_name + "/" + nthDir)
            for loc in loclist:
                copyfile(loc, dir_name + "/" + nthDir + "/" + loc)
            for script in scriptNames:
                copyfile(script, dir_name + "/" + nthDir + "/" + script)
            os.chdir(dir_name + "/" + nthDir) #adding an extra directory

            gc.load()

            run_node = gc.getNode('Run')
            
            parents = []

            if not parents:
                parents = None
            gc.createNodeVersion(run_node.nodeId, tag, parents)

            gc.commit()

            os.chdir('../')
            repo = git.Repo(os.getcwd())

            repo.index.add([nthDir,])

            repo.index.commit("incremental commit")
            tree = repo.tree()
            with open('.jarvis', 'w') as f:
                for obj in tree:
                    commithash = util.runProc("git log " + obj.path).replace('\n', ' ').split()[1]
                    if obj.path != '.jarvis':
                        f.write(obj.path + " " + commithash + "\n")
            repo.index.add(['.jarvis'])
            repo.index.commit('.jarvis commit')
            os.chdir('../')

    def __newCommit__(self):
        # BEHAVIOR LAYER NEXT

        # Reset visited for getLiteralsAttached graph traversal
        self.xp_state.visited = []
        experimentName = self.xp_state.EXPERIMENT_NAME
        experimentDirectory = self.xp_state.versioningDirectory + '/' + experimentName

        xpnv = ag.newExperimentVersion(self.xp_state)
        assert xpnv is not None

        literalsAttachedNames = self.__getLiteralsAttached__()

        original_dir  = os.getcwd()
        os.chdir(experimentDirectory)
        trialDirs = [x for x in os.listdir() if util.isNumber(x)]

        for trialDir in trialDirs:
            os.chdir(str(trialDir))
            with open('.' + experimentName + '.jarvis', 'r') as fp:
                config = json.load(fp)
            literals = {}
            for name in literalsAttachedNames:
                assert name in config
                literals[name] = config[name]
            trnv = ag.newTrialVersion(self.xp_state)
            assert trnv is not None
            ag.newExperimentTrialEdgeVersion(self.xp_state, xpnv, trnv)
            for litName in literals:
                ltnv = ag.newLiteralVersion(self.xp_state, litName, literals[litName])
                assert ltnv is not None
                ag.newTrialLiteralEdgeVersion(self.xp_state, trnv, ltnv)
            for artifactName in set(self.loclist + self.scriptNames) - self.xp_state.ghostFiles:
                rtnv = ag.newArtifactVersion(self.xp_state, artifactName, util.md5(artifactName))
                assert rtnv is not None
                ag.newTrialArtifactEdgeVersion(self.xp_state, trnv, rtnv)
            # We sleep to allow Ground server to complete all insertions before next iteration
            time.sleep(5)


            os.chdir('../')

        os.chdir(original_dir)

    def __getLiteralsAttached__(self):
        literalsAttachedNames = []
        if not self.parent:
            return literalsAttachedNames
        self.parent.__getLiteralsAttached__(literalsAttachedNames)
        return literalsAttachedNames


    def __pull__(self):
        """
        Partially refactored
        :return:
        """
        self.xp_state.visited = []
        driverfile = self.xp_state.jarvisFile

        if not util.isOrphan(self):
            self.loclist = list(map(lambda x: x.getLocation(), self.parent.out_artifacts))
        else:
            self.loclist = [self.getLocation(),]
        self.scriptNames = []
        if not util.isOrphan(self):
            self.parent.__run__(self.loclist, self.scriptNames)
        self.loclist = list(set(self.loclist))
        self.scriptNames = list(set(self.scriptNames))


        # Need to sort to compare
        self.loclist.sort()
        self.scriptNames.sort()

    def parallelPull(self, manifest={}):
        self.xp_state.versioningDirectory = os.path.expanduser('~') + '/' + 'jarvis.d'

        tmpexperiment = self.xp_state.tmpexperiment
        if os.path.exists(tmpexperiment):
            rmtree(tmpexperiment)
        os.mkdir(tmpexperiment)

        self.xp_state.visited = []

        if not util.isOrphan(self):
            self.loclist = list(map(lambda x: x.getLocation(), self.parent.out_artifacts))
        else:
            self.loclist = [self.getLocation(), ]
        self.scriptNames = []

        literalsAttached = set([])
        lambdas = []

        if not util.isOrphan(self):
            self.parent.__serialize__(lambdas, self.loclist, self.scriptNames)

        self.loclist = list(set(self.loclist))
        self.scriptNames = list(set(self.scriptNames)) 

        self.loclist.sort()
        self.scriptNames.sort()  

        for _, names in lambdas:
            literalsAttached |= set(names)

        original_dir = os.getcwd()
        experimentName = self.xp_state.jarvisFile.split('.')[0]  
        numTrials = 1
        literals = []
        literalNames = []
        config = {}

        for kee in self.xp_state.literalNameToObj:
            if kee in literalsAttached:
                config[kee] = self.xp_state.literalNameToObj[kee].v
                if self.xp_state.literalNameToObj[kee].__oneByOne__:
                    numTrials *= len(self.xp_state.literalNameToObj[kee].v)
                    literals.append(self.xp_state.literalNameToObj[kee].v)
                else:
                    if type(self.xp_state.literalNameToObj[kee].v) == tuple:
                        literals.append((self.xp_state.literalNameToObj[kee].v,))
                    else:
                        literals.append([self.xp_state.literalNameToObj[kee].v, ])
                literalNames.append(kee)

        literals = list(itertools.product(*literals))

        for i in range(numTrials):
            dst = tmpexperiment + '/' + str(i)
            copytree(os.getcwd(), dst, True)  

        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
        self.xp_state.ray['literalNames'] = literalNames  

        config['6zax7937'] = literals
        config['8ilk9274'] = literalNames
        
        @ray.remote
        def helperChangeDir(dir_path, lambdas, literals, config):
            os.chdir(dir_path)
            i = 0
            for f, names in lambdas:
                for each in names:
                    if i < len(literals):
                        config[each] = literals[i]
                        i += 1
                literal = list(map(lambda x: config[x], names))
                f(literal) 

            with open('.' + experimentName + '.jarvis', 'w') as fp:
                json.dump(config, fp)

        remaining_ids = []

        for i in range(numTrials):
            dir_path = tmpexperiment + '/' + str(i)
            remaining_ids.append(helperChangeDir.remote(dir_path, lambdas, literals[i], config))

        _, _ = ray.wait(remaining_ids, num_returns=numTrials)

        if not os.path.isdir(self.xp_state.versioningDirectory):
            os.mkdir(self.xp_state.versioningDirectory)

        moveBackFlag = False

        if os.path.exists(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME):
            move(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME + '/.git', '/tmp/')
            rmtree(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)
            moveBackFlag = True

        if manifest:
            os.chdir(tmpexperiment)

            dirs = [x for x in os.listdir() if util.isNumber(x)]
            dirs.sort()
            table_full = []
            table_small = []

            for trial in dirs:
                os.chdir(trial)
                with open('.' + experimentName + '.jarvis', 'r') as fp: #error here
                    config = json.load(fp)
                record_full = {}
                record_small = {}

                for literalName in literalNames:
                    record_full[literalName] = config[literalName]
                    record_small[literalName] = config[literalName]

                for artifactLabel in manifest:
                    record_full[artifactLabel] = util.loadArtifact(manifest[artifactLabel].loc)
                    if total_size(record_full[artifactLabel]) >= 1000:
                        record_small[artifactLabel] = " . . . "
                    else:
                        record_small[artifactLabel] = record_full[artifactLabel]
                    if util.isNumber(record_full[artifactLabel]):
                        record_full[artifactLabel] = eval(record_full[artifactLabel])
                    if util.isNumber(record_small[artifactLabel]):
                        record_small[artifactLabel] = eval(record_small[artifactLabel])
                record_small['__trialNum__'] = trial
                record_full['__trialNum__'] = trial

                table_full.append(record_full)
                table_small.append(record_small)
                os.chdir('../')

            df = pd.DataFrame(table_small)
            util.pickleTo(df, experimentName + '.pkl')

            os.chdir(original_dir)

        copytree(tmpexperiment, self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)

        os.chdir(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)
        if moveBackFlag:
            move('/tmp/.git', self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME)
            repo = git.Repo(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit('incremental commit')
        else:
            repo = git.Repo.init(os.getcwd())
            repo.git.add(A=True)
            repo.index.commit('initial commit')
        os.chdir(original_dir)

        self.__newCommit__()

        if manifest:
            return pd.DataFrame(table_full)



    def pull(self):

        # temporary fix for backward compatibility
        self.xp_state.versioningDirectory = 'jarvis.d'
        
        util.activate(self)
        userDefFiles = set(os.listdir()) - self.xp_state.ghostFiles
        try:
            while True:
                self.__pull__()
                # self.__commit__() COMMENTING OUT FOR GROUND REF PROTO
                subtreeMaxed = util.master_pop(self.xp_state.literals)
                if subtreeMaxed:
                    break
        except Exception as e:
            try:
                intermediateFiles = set(self.loclist) - userDefFiles
                for file in intermediateFiles:
                    if os.path.exists(file):
                        os.remove(file)
            except Exception as ee:
                print(ee)
            self.xp_state.literals = []
            self.xp_state.ghostFiles = set([])
            raise e

        intermediateFiles = set(self.loclist) - userDefFiles
        for file in intermediateFiles:
            os.remove(file)
        commitables = []
        for file in (userDefFiles & (set(self.loclist) | set(self.scriptNames))):
            copyfile(file, self.xp_state.versioningDirectory + '/' + file)
            commitables.append(file)

        os.chdir(self.xp_state.versioningDirectory)
        repo = git.Repo(os.getcwd())
        repo.index.add(commitables)
        repo.index.commit("incremental commit")
        tree = repo.tree()
        with open('.jarvis', 'w') as f:
            for obj in tree:
                commithash = util.runProc("git log " + obj.path).replace('\n', ' ').split()[1]
                if obj.path != '.jarvis':
                    f.write(obj.path + " " + commithash + "\n")
        repo.index.add(['.jarvis'])
        repo.index.commit('.jarvis commit')
        os.chdir('../')
        self.xp_state.literals = []
        self.xp_state.ghostFiles = set([])

        # self.__newCommit__()

    def peek(self, func = lambda x: x):
        trueVersioningDir = self.xp_state.versioningDirectory
        self.xp_state.versioningDirectory = '1fdf8583bfd663e98918dea393e273cc'
        try:
            self.pull()
            os.chdir(self.xp_state.versioningDirectory)
            listdir = [x for x in filter(util.isNumber, os.listdir())]
            _dir = str(len(listdir))
            if util.isPickle(self.loc):
                out = func(util.unpickle(_dir + '/' + self.loc))
            else:
                with open(_dir + '/' + self.loc, 'r') as f:
                    out = func(f.readlines())
            os.chdir('../')
        except Exception as e:
            out = e
        try:
            rmtree(self.xp_state.versioningDirectory)
        except:
            pass
        self.xp_state.versioningDirectory = trueVersioningDir
        return out

    def plot(self, rankdir=None):
        # WARNING: can't plot before pulling.
        # Prep globals, passed through arguments

        self.xp_state.nodes = {}
        self.xp_state.edges = []

        dot = Digraph()
        diagram = {"dot": dot, "counter": 0, "sha": {}}

        # with open('jarvis.d/.jarvis') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=' ')
        #     for row in reader:
        #         ob, sha = row
        #         diagram["sha"][ob] = sha

        if not util.isOrphan(self):
            self.parent.__plotWalk__(diagram)
        else:
            node_diagram_id = str(diagram["counter"])
            dot.node(node_diagram_id, self.loc, shape="box")
            self.xp_state.nodes[self.loc] = node_diagram_id


        dot.format = 'png'
        if rankdir == 'LR':
            dot.attr(rankdir='LR')
        dot.render('driver.gv', view=True)
        # return self.xp_state.edges

    def getLocation(self):
        return self.loc