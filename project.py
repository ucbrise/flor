#!/usr/bin/env python3
import os
import git
import inspect
import subprocess
import pickle
import itertools
import json
import datetime
import pandas as pd

from .ground import GroundClient
from graphviz import Digraph
from shutil import copyfile
from shutil import rmtree
from shutil import copytree

from ray.tune import register_trainable
from ray.tune import grid_search
from ray.tune import run_experiments

def func(lambdah):
    if Util.jarvisFileIsIpynb():
        filename =  Util.jarvisFile
    else:
        filename = inspect.getsourcefile(lambdah).split('/')[-1]
    def wrapped_func(in_artifacts, out_artifacts):
        if in_artifacts:
            in_args = []
            for in_art in [in_art.loc if Util.isJarvisClass(in_art) else in_art for in_art in in_artifacts]:
                if Util.isPickle(in_art):
                    x = Util.unpickle(in_art)
                elif Util.isCsv(in_art):
                    x = in_art
                elif Util.isLoc(in_art):
                    with open(in_art, 'r') as f:
                        x = [i.strip() for i in f.readlines() if i.strip()]
                    if len(x) == 1:
                        x = x[0]
                else:
                    x = in_art
                in_args.append(x)
            outs = lambdah(*in_args)
        else:
            outs = lambdah()
        if Util.isIterable(outs):
            try:
                assert len(outs) == len(out_artifacts)
                for out, out_loc in zip(outs, [out_art.loc for out_art in out_artifacts]):
                    if Util.isPickle(out_loc):
                        with open(out_loc, 'wb') as f:
                            pickle.dump(out, f)
                    else:
                        with open(out_loc, 'w') as f:
                            if Util.isIterable(out):
                                for o in out:
                                    f.write(str(o) + '\n')
                            else:
                                f.write(str(out) + '\n')
            except:
                assert len(out_artifacts) == 1
                outs = [outs,]
                for out, out_loc in zip(outs, [out_art.loc for out_art in out_artifacts]):
                    if Util.isPickle(out_loc):
                        with open(out_loc, 'wb') as f:
                            pickle.dump(out, f)
                    else:
                        with open(out_loc, 'w') as f:
                            if Util.isIterable(out):
                                for o in out:
                                    f.write(str(o) + '\n')
                            else:
                                f.write(str(out) + '\n')
        elif out_artifacts and outs is not None:
            out_loc = out_artifacts[0].loc
            if Util.isPickle(out_loc):
                with open(out_loc, 'wb') as f:
                    pickle.dump(outs, f)
            else:
                with open(out_loc, 'w') as f:
                    if Util.isIterable(outs):
                        for o in outs:
                            f.write(str(o) + '\n')
                    else:
                        f.write(str(outs) + '\n')
        else:
            raise AssertionError("Missing location to write or Missing return value.")
        return lambdah.__name__
    return filename, lambdah.__name__, wrapped_func

def groundClient(backend):
    Util.gc = GroundClient(backend)

def jarvisFile(loc):
    Util.jarvisFile = loc

class Literal:

    def __init__(self, v, name=None):
        """

        :param v: Literal value
        :param name: must be globally unique per experiment
        """
        self.v = v
        self.loc = 'ghost_literal_' + str(Util.literalFilenamesAndIncr()) + '.pkl'
        self.__oneByOne__ = False
        self.i = 0
        self.n = 1

        # The name is used in the visualization and Ground versioning
        if name is None:
            temp = self.loc.split('.')[0]
            i = int(temp.split('_')[2])
            self.name = 'ghost' + str(i)
            while self.name in Util.literalNames:
                i += 1
                self.name = 'ghost' + str(i)
        else:
            self.name = name
            assert name not in Util.literalNames
        Util.literalNames |= {self.name}

        Util.literalNameToObj[self.name] = self

    def forEach(self):
        if not Util.isIterable(self.v):
            raise TypeError("Cannot iterate over literal {}".format(self.v))
        self.__oneByOne__ = True
        self.n = len(self.v)
        return self

    def getLocation(self):
        return self.loc

    def __pop__(self):
        if self.i >= self.n:
            return False
        if self.__oneByOne__:
            Util.pickleTo(self.v[self.i], self.loc)
        else:
            Util.pickleTo(self.v, self.loc)
        self.i += 1
        return True

    def __enable__(self):
        Util.ghostFiles |= {self.loc}
        Util.literals.append(self)
        self.__reset__()

    def __reset__(self):
        self.i = 0
        self.__pop__()

class Artifact:

    def __init__(self, loc, parent=None, manifest=False):
        self.loc = loc
        self.parent = parent

        if self.parent:
            self.parent.out_artifacts.append(self)

    def __commit__(self):

        gc = Util.gc
        dir_name = Util.versioningDirectory
        loclist = self.loclist
        scriptNames = self.scriptNames
        tag = {
            'Artifacts': [i for i in loclist],
            'Actions': [i for i in scriptNames]
        }

        for literal in Util.literals:
            if literal.name:
                try:
                    value = str(Util.unpickle(literal.loc))
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
                    commithash = Util.runProc("git log " + obj.path).replace('\n', ' ').split()[1]
                    if obj.path != '.jarvis':
                        f.write(obj.path + " " + commithash + "\n")
            repo.index.add(['.jarvis'])
            repo.index.commit('.jarvis commit')
            os.chdir('../')
        else:

            listdir = [x for x in filter(Util.isNumber, os.listdir(dir_name))]

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
                    commithash = Util.runProc("git log " + obj.path).replace('\n', ' ').split()[1]
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
        Util.visited = []
        driverfile = Util.jarvisFile

        if not Util.isOrphan(self):
            self.loclist = list(map(lambda x: x.getLocation(), self.parent.out_artifacts))
        else:
            self.loclist = [self.getLocation(),]
        self.scriptNames = []
        if not Util.isOrphan(self):
            self.parent.__run__(self.loclist, self.scriptNames)
        self.loclist = list(set(self.loclist))
        self.scriptNames = list(set(self.scriptNames))


        # Need to sort to compare
        self.loclist.sort()
        self.scriptNames.sort()

    def parallelPull(self, manifest={}):

        Util.versioningDirectory = os.path.expanduser('~') + '/' + 'jarvis.d'

        # Runs one experiment per pull
        # Each experiment has many trials

        tmpexperiment = '/tmp/de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3'
        if os.path.exists(tmpexperiment):
            rmtree(tmpexperiment)
            os.mkdir(tmpexperiment)
        else:
            os.mkdir(tmpexperiment)

        Util.visited = []

        if not Util.isOrphan(self):
            self.loclist = list(map(lambda x: x.getLocation(), self.parent.out_artifacts))
        else:
            self.loclist = [self.getLocation(),]
        self.scriptNames = []

        literalsAttached = set([])
        lambdas = []
        if not Util.isOrphan(self):
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
        for kee in Util.literalNameToObj:
            if kee in literalsAttached:
                if Util.literalNameToObj[kee].__oneByOne__:
                    config[kee] = grid_search(Util.literalNameToObj[kee].v)
                    numTrials *= len(Util.literalNameToObj[kee].v)
                    literals.append(Util.literalNameToObj[kee].v)
                else:
                    config[kee] = Util.literalNameToObj[kee].v
                    if Util.isIterable(Util.literalNameToObj[kee].v):
                        if type(Util.literalNameToObj[kee].v) == tuple:
                            literals.append((Util.literalNameToObj[kee].v,))
                        else:
                            literals.append([Util.literalNameToObj[kee].v,])
                literalNames.append(kee)

        literals = list(itertools.product(*literals))
        config['6zax7937'] = literals
        config['8ilk9274'] = literalNames

        for i in range(numTrials):
            dst = tmpexperiment + '/' + str(i)
            copytree(os.getcwd(), dst, True)


        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        register_trainable('exportedExec', exportedExec)

        experimentName = Util.jarvisFile.split('.')[0]

        run_experiments({
            experimentName : {
                'run': 'exportedExec',
                'resources': {'cpu': 1, 'gpu': 0},
                'config': config
            }
        })

        if not os.path.isdir(Util.versioningDirectory):
            os.mkdir(Util.versioningDirectory)
        copytree(tmpexperiment, Util.versioningDirectory + '/' + Util.jarvisFile.split('.')[0] + '_' + ts)

        if manifest:

            os.chdir(tmpexperiment)

            dirs = [x for x in os.listdir() if Util.isNumber(x)]
            table = []

            for trial in dirs:
                os.chdir(trial)
                with open('.' + experimentName + '.jarvis', 'r') as fp:
                    config = json.load(fp)
                record = {}
                for literalName in literalNames:
                    record[literalName] = config[literalName]
                for artifactLabel in manifest:
                    record[artifactLabel] = Util.loadArtifact(manifest[artifactLabel].loc)
                    if Util.isNumber(record[artifactLabel]):
                        record[artifactLabel] = eval(record[artifactLabel])
                table.append(record)
                os.chdir('../')

            os.chdir(original_dir)

            return pd.DataFrame(table)


    def pull(self):

        # temporary fix for backward compatibility
        Util.versioningDirectory = 'jarvis.d'

        Util.activate(self)
        userDefFiles = set(os.listdir()) - Util.ghostFiles
        try:
            while True:
                self.__pull__()
                self.__commit__()
                subtreeMaxed = Util.master_pop(Util.literals)
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
            Util.literals = []
            Util.ghostFiles = set([])
            raise e
        intermediateFiles = set(self.loclist) - userDefFiles
        for file in intermediateFiles:
            os.remove(file)
        commitables = []
        for file in (userDefFiles & (set(self.loclist) | set(self.scriptNames))):
            copyfile(file, Util.versioningDirectory + '/' + file)
            commitables.append(file)
        os.chdir(Util.versioningDirectory)
        repo = git.Repo(os.getcwd())
        repo.index.add(commitables)
        repo.index.commit("incremental commit")
        tree = repo.tree()
        with open('.jarvis', 'w') as f:
            for obj in tree:
                commithash = Util.runProc("git log " + obj.path).replace('\n', ' ').split()[1]
                if obj.path != '.jarvis':
                    f.write(obj.path + " " + commithash + "\n")
        repo.index.add(['.jarvis'])
        repo.index.commit('.jarvis commit')
        os.chdir('../')
        Util.literals = []
        Util.ghostFiles = set([])

    def peek(self, func = lambda x: x):
        trueVersioningDir = Util.versioningDirectory
        Util.versioningDirectory = '1fdf8583bfd663e98918dea393e273cc'
        try:
            self.pull()
            os.chdir(Util.versioningDirectory)
            listdir = [x for x in filter(Util.isNumber, os.listdir())]
            dir = str(len(listdir))
            if Util.isPickle(self.loc):
                out = func(Util.unpickle(dir + '/' + self.loc))
            else:
                with open(dir + '/' + self.loc, 'r') as f:
                    out = func(f.readlines())
            os.chdir('../')
        except Exception as e:
            out = e
        try:
            rmtree(Util.versioningDirectory)
        except:
            pass
        Util.versioningDirectory = trueVersioningDir
        return out

    def plot(self, rankdir=None):
        # WARNING: can't plot before pulling.
        # Prep globals, passed through arguments

        Util.nodes = {}
        Util.edges = []

        dot = Digraph()
        diagram = {"dot": dot, "counter": 0, "sha": {}}

        # with open('jarvis.d/.jarvis') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=' ')
        #     for row in reader:
        #         ob, sha = row
        #         diagram["sha"][ob] = sha

        if not Util.isOrphan(self):
            self.parent.__plotWalk__(diagram)
        else:
            node_diagram_id = str(diagram["counter"])
            dot.node(node_diagram_id, self.loc, shape="box")
            Util.nodes[self.loc] = node_diagram_id


        dot.format = 'png'
        if rankdir == 'LR':
            dot.attr(rankdir='LR')
        dot.render('driver.gv', view=True)
        return Util.edges

    def getLocation(self):
        return self.loc

class Action:

    def __init__(self, func, in_artifacts=None):
        self.filenameWithFunc, self.funcName, self.func = func
        self.out_artifacts = []

        if in_artifacts:
            temp_artifacts = []
            for in_art in in_artifacts:
                if not Util.isJarvisClass(in_art):
                    if Util.isIterable(in_art):
                        in_art = Literal(in_art)
                        in_art.forEach()
                    else:
                        in_art = Literal(in_art)
                temp_artifacts.append(in_art)
            in_artifacts = temp_artifacts

        self.in_artifacts = in_artifacts

    def __run__(self, loclist, scriptNames):
        outNames = ''
        for out_artifact in self.out_artifacts:
            outNames += out_artifact.loc
        if self.funcName + outNames in Util.visited:
            return
        scriptNames.append(self.filenameWithFunc)
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                loclist.append(artifact.loc)
                if not Util.isOrphan(artifact):
                    artifact.parent.__run__(loclist, scriptNames)
        self.func(self.in_artifacts, self.out_artifacts)
        Util.visited.append(self.funcName + outNames)

    def __serialize__(self, lambdas, loclist, scriptNames):
        outNames = ''
        namedLiterals = []
        for out_artifact in self.out_artifacts:
            outNames += out_artifact.loc
        if self.funcName + outNames in Util.visited:
            return
        scriptNames.append(self.filenameWithFunc)
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                loclist.append(artifact.loc)
                if not Util.isOrphan(artifact):
                    artifact.parent.__serialize__(lambdas, loclist, scriptNames)
                elif type(artifact) == Literal:
                    namedLiterals.append(artifact.name)

        def _lambda(literals=[]):
            i = 0
            args = []
            for in_art in self.in_artifacts:
                if type(in_art) == Literal:
                    args.append(literals[i])
                    i += 1
                else:
                    args.append(in_art)
            self.func(args, self.out_artifacts)

        lambdas.append((_lambda, namedLiterals))
        Util.visited.append(self.funcName + outNames)


    def __plotWalk__(self, diagram):
        dot = diagram["dot"]

        # Create nodes for the children

        to_list = []

        # Prepare the children nodes
        for child in self.out_artifacts:
            node_diagram_id = str(diagram["counter"])
            dot.node(node_diagram_id, child.loc, shape="box")
            Util.nodes[child.loc] = node_diagram_id
            to_list.append((node_diagram_id, child.loc))
            diagram["counter"] += 1

        # Prepare this node
        node_diagram_id = str(diagram["counter"])
        dot.node(node_diagram_id, self.funcName, shape="ellipse")
        Util.nodes[self.funcName] = node_diagram_id
        diagram["counter"] += 1

        # Add the script artifact
        node_diagram_id_script = str(diagram["counter"])
        dot.node(node_diagram_id_script, self.filenameWithFunc, shape="box")
        diagram["counter"] += 1
        dot.edge(node_diagram_id_script, node_diagram_id)
        Util.edges.append((node_diagram_id_script, node_diagram_id))

        for to_node, loc in to_list:
            dot.edge(node_diagram_id, to_node)
            Util.edges.append((node_diagram_id, to_node))

        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.getLocation() in Util.nodes:
                    if (Util.nodes[artifact.getLocation()], node_diagram_id) not in Util.edges:
                        dot.edge(Util.nodes[artifact.getLocation()], node_diagram_id)
                        Util.edges.append((Util.nodes[artifact.getLocation()], node_diagram_id))
                else:
                    if not Util.isOrphan(artifact):
                        from_nodes = artifact.parent.__plotWalk__(diagram)
                        for from_node, loc in from_nodes:
                            if loc in [art.getLocation() for art in self.in_artifacts]:
                                if (from_node, node_diagram_id) not in Util.edges:
                                    dot.edge(from_node, node_diagram_id)
                                    Util.edges.append((from_node, node_diagram_id))
                    else:
                        node_diagram_id2 = str(diagram["counter"])
                        if type(artifact) == Literal and artifact.name:
                            dot.node(node_diagram_id2, artifact.name,
                                     shape="box")
                        else:
                            dot.node(node_diagram_id2, artifact.loc,
                                     shape="box")
                        Util.nodes[artifact.loc] = node_diagram_id2
                        diagram["counter"] += 1
                        if (node_diagram_id2, node_diagram_id) not in Util.edges:
                            dot.edge(node_diagram_id2, node_diagram_id)
                            Util.edges.append((node_diagram_id2, node_diagram_id))

        return to_list

class Util:

    # The following are experiment-wide variables

    edges = []
    gc = None
    ghostFiles = set([])
    jarvisFile = 'driver.py'
    literals = []
    literalFilenames = 0
    literalNames = set([])
    literalNameToObj = {}
    nodes = {}
    versioningDirectory = os.path.expanduser('~') + '/' + 'jarvis.d'
    visited = []

    @staticmethod
    def isLoc(loc):
        try:
            ext = loc.split('.')[1]
            return True
        except:
            return False
    @staticmethod
    def isJarvisClass(obj):
        return type(obj) == Artifact or type(obj) == Action or type(obj) == Literal
    @staticmethod
    def isPickle(loc):
        try:
            return loc.split('.')[1] == 'pkl'
        except:
            return False
    @staticmethod
    def isCsv(loc):
        try:
            return loc.split('.')[1] == 'csv'
        except:
            return False
    @staticmethod
    def jarvisFileIsIpynb():
        return Util.jarvisFile.split('.')[1] == 'ipynb'
    @staticmethod
    def isIterable(obj):
        return type(obj) == list or type(obj) == tuple
    @staticmethod
    def runProc(bashCommand):
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return str(output, 'UTF-8')
    @staticmethod
    def pickleTo(obj, loc):
        with open(loc, 'wb') as f:
            pickle.dump(obj, f)
    @staticmethod
    def unpickle(loc):
        with open(loc, 'rb') as f:
            x = pickle.load(f)
        return x
    @staticmethod
    def literalFilenamesAndIncr():
        x = Util.literalFilenames
        Util.literalFilenames += 1
        return x
    @staticmethod
    def isOrphan(obj):
        return type(obj) == Literal or (type(obj) == Artifact and obj.parent is None)
    @staticmethod
    def isNumber(s):
        if type(s) == int or type(s) == float:
            return True
        try:
            float(s)
            return True
        except:
            return False
    @staticmethod
    def master_pop(literals):
        if not literals:
            return True
        subtreeMaxed = Util.master_pop(literals[0:-1])
        if subtreeMaxed:
            popSuccess = literals[-1].__pop__()
            if not popSuccess:
                return True
            [literal.__reset__() for literal in literals[0:-1]]
        return False
    @staticmethod
    def activate(pseudoArtifact):
        if type(pseudoArtifact) == Literal:
            pseudoArtifact.__enable__()
        elif not Util.isOrphan(pseudoArtifact) and pseudoArtifact.parent.in_artifacts:
            for in_art in pseudoArtifact.parent.in_artifacts:
                Util.activate(in_art)
    @staticmethod
    def loadArtifact(loc):
        if Util.isPickle(loc):
            x = Util.unpickle(loc)
        elif Util.isLoc(loc):
            with open(loc, 'r') as f:
                x = [i.strip() for i in f.readlines() if i.strip()]
            if len(x) == 1:
                x = x[0]
        else:
            raise NotImplementedError("Don't know how to load that file.")
        return x

