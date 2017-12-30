#!/usr/bin/env python3

import os
import git
import json
import itertools
import datetime
import pandas as pd

from graphviz import Digraph
from shutil import copyfile
from shutil import rmtree
from shutil import copytree

from ray.tune import register_trainable
from ray.tune import grid_search
from ray.tune import run_experiments

from .. import util

class Artifact:

    def __init__(self, loc, parent, manifest, xp_state):
        self.loc = loc
        self.parent = parent

        if self.parent:
            self.parent.out_artifacts.append(self)

        self.xp_state = xp_state

    def __commit__(self):

        gc = self.xp_state.gc
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
            # Move new files to the artifacts repo
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
            os.chdir(dir_name + "/" + nthDir)

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

        # Runs one experiment per pull
        # Each experiment has many trials

        tmpexperiment = '/tmp/de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3'
        if os.path.exists(tmpexperiment):
            rmtree(tmpexperiment)
            os.mkdir(tmpexperiment)
        else:
            os.mkdir(tmpexperiment)

        self.xp_state.visited = []

        if not util.isOrphan(self):
            self.loclist = list(map(lambda x: x.getLocation(), self.parent.out_artifacts))
        else:
            self.loclist = [self.getLocation(),]
        self.scriptNames = []

        literalsAttached = set([])
        lambdas = []
        if not util.isOrphan(self):
            self.parent.__serialize__(lambdas, self.loclist, self.scriptNames)

        self.loclist = list(set(self.loclist))
        self.scriptNames = list(set(self.scriptNames))

        # Need to sort to compare
        self.loclist.sort()
        self.scriptNames.sort()

        for _, names in lambdas:
            literalsAttached |= set(names)

        original_dir = os.getcwd()
        def exportedExec(config, reporter):
            tee = tuple([])
            for litName in config['8ilk9274']:
                tee += (config[litName], )
            i = -1
            for j, v in enumerate(config['6zax7937']):
                if v == tee:
                    i = j
                    break
            assert i >= 0
            os.chdir(tmpexperiment + '/' + str(i))
            for f, names in lambdas:
                literals = list(map(lambda x: config[x], names))
                f(literals)
            reporter(timesteps_total=1)
            with open('.' + experimentName + '.jarvis', 'w') as fp:
                json.dump(config, fp)
            os.chdir(original_dir)

        config = {}
        numTrials = 1
        literals = []
        literalNames = []
        for kee in self.xp_state.literalNameToObj:
            if kee in literalsAttached:
                if self.xp_state.literalNameToObj[kee].__oneByOne__:
                    config[kee] = grid_search(self.xp_state.literalNameToObj[kee].v)
                    numTrials *= len(self.xp_state.literalNameToObj[kee].v)
                    literals.append(self.xp_state.literalNameToObj[kee].v)
                else:
                    config[kee] = self.xp_state.literalNameToObj[kee].v
                    if util.isIterable(self.xp_state.literalNameToObj[kee].v):
                        if type(self.xp_state.literalNameToObj[kee].v) == tuple:
                            literals.append((self.xp_state.literalNameToObj[kee].v, ))
                        else:
                            literals.append([self.xp_state.literalNameToObj[kee].v, ])
                    else:
                        literals.append([self.xp_state.literalNameToObj[kee].v, ])
                literalNames.append(kee)

        literals = list(itertools.product(*literals))
        config['6zax7937'] = literals
        config['8ilk9274'] = literalNames

        for i in range(numTrials):
            dst = tmpexperiment + '/' + str(i)
            copytree(os.getcwd(), dst, True)


        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        register_trainable('exportedExec', exportedExec)

        experimentName = self.xp_state.jarvisFile.split('.')[0]

        run_experiments({
            experimentName : {
                'run': 'exportedExec',
                'resources': {'cpu': 1, 'gpu': 0},
                'config': config
            }
        })

        if not os.path.isdir(self.xp_state.versioningDirectory):
            os.mkdir(self.xp_state.versioningDirectory)
        copytree(tmpexperiment, self.xp_state.versioningDirectory + '/' + self.xp_state.jarvisFile.split('.')[0] + '_' + ts)

        if manifest:

            os.chdir(tmpexperiment)

            dirs = [x for x in os.listdir() if util.isNumber(x)]
            table = []

            for trial in dirs:
                os.chdir(trial)
                with open('.' + experimentName + '.jarvis', 'r') as fp:
                    config = json.load(fp)
                record = {}
                for literalName in literalNames:
                    record[literalName] = config[literalName]
                for artifactLabel in manifest:
                    record[artifactLabel] = util.loadArtifact(manifest[artifactLabel].loc)
                    if util.isNumber(record[artifactLabel]):
                        record[artifactLabel] = eval(record[artifactLabel])
                table.append(record)
                os.chdir('../')

            os.chdir(original_dir)

            return pd.DataFrame(table)


    def pull(self):

        # temporary fix for backward compatibility
        self.xp_state.versioningDirectory = 'jarvis.d'

        util.activate(self)
        userDefFiles = set(os.listdir()) - self.xp_state.ghostFiles
        try:
            while True:
                self.__pull__()
                self.__commit__()
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
