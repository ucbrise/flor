#!/usr/bin/env python3
import os, sys, git, subprocess, csv, inspect
from ground import GroundClient
from graphviz import Digraph
from shutil import copyfile
from random import shuffle
import networkx as nx
import pickle
import math


def __run_proc__(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return str(output, 'UTF-8')

def func(f):
    def name_func():
        return inspect.getsourcefile(f).split('/')[-1]
    def wrapped_func(in_artifacts, out_artifacts):
        f(in_artifacts, out_artifacts)
        return inspect.getsourcefile(f).split('/')[-1]
    return [name_func, wrapped_func]


def ground_client(backend):
    global __gc__
    __gc__ = GroundClient(backend)

def jarvisFile(loc):
    global __jarvisFile__
    __jarvisFile__ = loc

class Sample:

    """
    Constraint: Can only sample static, pre-existing data.
    """

    def __init__(self, rate, loc, batch, times=1, to_csv=False):
        assert rate <= 1 and rate > 0
        assert times >= 1

        global __sample_interm_files__

        self.rate = rate
        self.times = times
        self.batch = batch
        self.to_csv = to_csv

        artifact = Artifact(loc)

        # Action part
        self.action = Action([self.__dummy__, self.__dummy__], [artifact])

        # Artifact part
        if not self.to_csv:
            self.loc = 'sampled_' + loc.split('.')[0] + '.pkl'
        else:
            self.loc = 'sampled_' + loc.split('.')[0] + '.csv'
        self.dir = 'jarvis.d'
        self.parent = self.action

        __sample_interm_files__ |= {self.loc,}

        self.parent.out_artifacts.append(self)

        self.popped = False
        self.superBuffer = None

        self.__sample__(loc, self.loc)

        __samples__.append(self)

        # Initialization pop
        self.__pop__()

    def __sample__(self, in_artifact, out_artifact):
        loc = in_artifact
        out_loc = out_artifact
        super_buffer = []
        with open(loc, 'r') as f:
            buffer = [x.strip() for x in f.readlines() if x.strip()]
        for i in range(self.times):
            shuffle(buffer)
            buff = buffer[0:math.ceil(self.rate*len(buffer))]
            super_buffer.append(buff)
        # with open(out_loc, 'wb') as f:
        #     pickle.dump(super_buffer, f)
        assert len(super_buffer) > 0
        self.superBuffer = super_buffer
        self.i = 0
        self.n = len(super_buffer)
        if not self.batch:
            self.j = 0
            self.m = len(super_buffer[0])
            assert self.m > 0
        return 'sample.79b14259b09e2608e3f704d0fd5a80fd'

    def __dummy__(self, in_artifacts=None, out_artifacts=None):
        return 'sample.79b14259b09e2608e3f704d0fd5a80fd'

    def __pop__(self):
        if self.i >= self.n:
            return False
        if self.batch:
            if not self.to_csv:
                with open(self.loc, 'wb') as f:
                    pickle.dump(self.superBuffer[self.i], f)
                    self.i += 1
            else:
                with open(self.loc, 'w') as f:
                    for line in self.superBuffer[self.i]:
                        f.write(line + '\n')
                    self.i += 1
            return True
        else:
            if self.j >= self.m:
                self.i += 1
                self.j = 0
                return self.__pop__()
            if not self.to_csv:
                with open(self.loc, 'wb') as f:
                    pickle.dump(self.superBuffer[self.i][self.j], f)
                    self.j += 1
            else:
                with open(self.loc, 'w') as f:
                    for line in self.superBuffer[self.i][self.j]:
                        f.write(line + '\n')
                    self.j += 1
            return True

    def __reset__(self):
        self.i = 0
        self.j = 0
        self.__pop__()

    def getLocation(self):
        return self.loc



class Artifact:

    # loc: location
    # parent: each artifact is produced by 1 action
    def __init__(self, loc, parent=None):
        self.loc = loc
        self.dir = "jarvis.d"
        self.parent = parent

        # Now we bind the artifact to its parent
        if self.parent:
            self.parent.out_artifacts.append(self)

        self.popped = False

    def __commit__(self):
        gc = __gc__
        dir_name = self.dir
        loclist = self.loclist
        scriptNames = self.scriptNames
        tag = {
            'Artifacts': [i for i in loclist],
            'Actions': [i for i in scriptNames]
        }
        if not os.path.exists(dir_name):
            nodeid = gc.createNode('Run')
            gc.createNodeVersion(nodeid, tag)

            # Create a node for every artifact (everything in jarvis.d)
            # Create a node version for every artifact (everything in jarvis.d)

            # Roughly, actions map to edges
            # Create an edge for every out-artifact for every action
            # Create an edge version for every out-artifact for every action

            # Create a graph
            # Create a graph version, pass it the edges created

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
                    commithash = __run_proc__("git log " + obj.path).replace('\n', ' ').split()[1]
                    if obj.path != '.jarvis':
                        f.write(obj.path + " " + commithash + "\n")
            repo.index.add(['.jarvis'])
            repo.index.commit('.jarvis commit')
            os.chdir('../')
        else:

            def is_number(s):
                try:
                    float(s)
                    return True
                except ValueError:
                    return False

            listdir = [x for x in filter(is_number, os.listdir(dir_name))]

            nthDir =  str(len(listdir) + 1)
            os.makedirs(dir_name + "/" + nthDir)
            for loc in loclist:
                copyfile(loc, dir_name + "/" + nthDir + "/" + loc)
            for script in scriptNames:
                copyfile(script, dir_name + "/" + nthDir + "/" + script)
            os.chdir(dir_name + "/" + nthDir)
            # repo = git.Repo.init(os.getcwd())

            gc.load()

            run_node = gc.getNode('Run')
            # run_node_latest_versions = gc.getNodeLatestVersions('Run')
            parents = []
            # for nlv in run_node_latest_versions:
            #     if nlv.tags == tag:
            #         parents.append(nlv.nodeVersionId)
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
                    commithash = __run_proc__("git log " + obj.path).replace('\n', ' ').split()[1]
                    if obj.path != '.jarvis':
                        f.write(obj.path + " " + commithash + "\n")
            repo.index.add(['.jarvis'])
            repo.index.commit('.jarvis commit')
            os.chdir('../')

    def __pop__(self, samples):
        if not samples:
            return True
        subtreeMaxed = self.__pop__(samples[0:-1])
        if subtreeMaxed:
            popSuccess = samples[-1].__pop__()
            if not popSuccess:
                return True
            [sample.__reset__() for sample in samples[0:-1]]
        return False


    def __pull__(self):
        global __visited__
        __visited__ = []

        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        driverfile = __jarvisFile__ #module.__file__.split('/')[-1]

        if self.parent:
            loclist = list(map(lambda x: x.getLocation(), self.parent.out_artifacts))
        else:
            loclist = [self.getLocation(),]
        if self.parent:
            self.parent.__run__(loclist)
        loclist = list(set(loclist))


        # get the script names
        scriptNames = [driverfile,]
        if self.parent:
            self.parent.__scriptNameWalk__(scriptNames)
        scriptNames = [x for x in set(scriptNames) if x != 'sample.79b14259b09e2608e3f704d0fd5a80fd']


        # Need to sort to compare
        loclist.sort()
        scriptNames.sort()

        self.loclist = loclist
        self.scriptNames = scriptNames



    def pull(self):
        userDefFiles = set(os.listdir()) - __sample_interm_files__
        while True:
            self.__pull__()
            self.__commit__()
            subtreeMaxed = self.__pop__(__samples__)
            if subtreeMaxed:
                break
        intermediateFiles = set(self.loclist) - userDefFiles
        for file in intermediateFiles:
            os.remove(file)
        commitables = []
        for file in (userDefFiles & (set(self.loclist) | set(self.scriptNames))):
            copyfile(file, self.dir + '/' + file)
            commitables.append(file)
        os.chdir(self.dir)
        repo = git.Repo(os.getcwd())
        repo.index.add(commitables)
        repo.index.commit("incremental commit")
        tree = repo.tree()
        with open('.jarvis', 'w') as f:
            for obj in tree:
                commithash = __run_proc__("git log " + obj.path).replace('\n', ' ').split()[1]
                if obj.path != '.jarvis':
                    f.write(obj.path + " " + commithash + "\n")
        repo.index.add(['.jarvis'])
        repo.index.commit('.jarvis commit')
        os.chdir('../')




    def plot(self, rankdir=None):
        # WARNING: can't plot before pulling.
        # Prep globals, passed through arguments
        global __nodes__
        __nodes__ = {}

        dot = Digraph()
        diagram = {"dot": dot, "counter": 0, "sha": {}}

        # with open('jarvis.d/.jarvis') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=' ')
        #     for row in reader:
        #         ob, sha = row
        #         diagram["sha"][ob] = sha

        if self.parent:
            self.parent.__plotWalk__(diagram)
        else:
            node_diagram_id = str(diagram["counter"])
            dot.node(node_diagram_id, self.loc, shape="box")
            __nodes__[self.loc] = node_diagram_id


        dot.format = 'png'
        if rankdir == 'LR':
            dot.attr(rankdir='LR')
        dot.render('driver.gv', view=True)

    def to_graph(self, rankdir=None):
        # WARNING: can't plot before pulling.
        # Prep globals, passed through arguments
        global __nodes__
        __nodes__ = {}


        diagram = {
            "g": nx.Graph(),
            "counter": 0,
            "sha": {}
            }

        with open('jarvis.d/.jarvis') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                ob, sha = row
                diagram["sha"][ob] = sha

        if self.parent:
            self.parent.__graphWalk__(diagram)
        else:
            raise NotImplementedError()

        return diagram['g']


    def getLocation(self):
        return self.loc

    def hasChanged(self):
        pass

    """
    We will want to check the loc prefix to decide where to look
    for existence e.g. http, s3, sftp, etc.
    No pre-defined prefix, assume local filesystem.
    """
    def exists(self):
        if not os.path.isfile(self.loc):
            print(self.loc + " not found.")
            sys.exit(1)

    """
    We assume an open-ended Integrity predicate on each artifact. 
    The design of this predicate language is TBD
    """
    def isLegal(self):
        pass

    def stat(self):
        pass


class Action:

    def __init__(self, func, in_artifacts=None):
        self.name = func[0]()
        self.func = func[1]
        self.out_artifacts = []
        self.in_artifacts = in_artifacts

    def __run__(self, loclist):
        outNames = ''
        for out_artifact in self.out_artifacts:
            outNames += out_artifact.getLocation()
        if self.name+outNames in __visited__:
            return
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                loclist.append(artifact.loc)
                if artifact.parent:
                    artifact.parent.__run__(loclist)
        self.script = self.func(self.in_artifacts, self.out_artifacts)
        __visited__.append(self.script+outNames)

    def produce(self, loc):
        return Artifact(loc, self)

    def __scriptNameWalk__(self, scriptNames):
        scriptNames.append(self.script)
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.parent:
                    artifact.parent.__scriptNameWalk__(scriptNames)

                
    def __plotWalk__(self, diagram):
        dot = diagram["dot"]
        
        # Create nodes for the children
        
        to_list = []
        
        # Prepare the children nodes
        for child in self.out_artifacts:
            node_diagram_id = str(diagram["counter"])
            dot.node(node_diagram_id, child.loc, shape="box")
            __nodes__[child.loc] = node_diagram_id
            to_list.append((node_diagram_id, child.loc))
            diagram["counter"] += 1
        
        # Prepare this node
        node_diagram_id = str(diagram["counter"])
        dot.node(node_diagram_id, self.script.split('.')[0], shape="ellipse")
        __nodes__[self.script] = node_diagram_id
        diagram["counter"] += 1
        
        # Add the script artifact
        if self.script != 'sample.79b14259b09e2608e3f704d0fd5a80fd':
            node_diagram_id_script = str(diagram["counter"])
            dot.node(node_diagram_id_script, self.script, shape="box")
            diagram["counter"] += 1
            dot.edge(node_diagram_id_script, node_diagram_id)

        for to_node, loc in to_list:
            dot.edge(node_diagram_id, to_node)
        
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.getLocation() in __nodes__:
                    dot.edge(__nodes__[artifact.getLocation()], node_diagram_id)
                else:
                    if artifact.parent:
                        from_nodes = artifact.parent.__plotWalk__(diagram)
                        for from_node, loc in from_nodes:
                            if loc in [art.getLocation() for art in self.in_artifacts]:
                                dot.edge(from_node, node_diagram_id)
                    else:
                        node_diagram_id2 = str(diagram["counter"])
                        dot.node(node_diagram_id2, artifact.loc,
                                 shape="box")
                        __nodes__[artifact.loc] = node_diagram_id2
                        diagram["counter"] += 1
                        dot.edge(node_diagram_id2, node_diagram_id)

        
        return to_list


    def __graphWalk__(self, diagram):
        g = diagram['g']
        # Create nodes for the children
        
        to_list = []
        
        # Prepare the children nodes
        for child in self.out_artifacts:
            node_diagram_id = str(diagram["counter"])
            g.add_node(node_diagram_id, 
                {
                    "text": child.loc ,
                    "shape": "box"
                })
            __nodes__[child.loc] = node_diagram_id
            to_list.append((node_diagram_id, child.loc))
            diagram["counter"] += 1
        
        # Prepare this node
        node_diagram_id = str(diagram["counter"])
        g.add_node(node_diagram_id, 
            {
                    "text": self.script.split('.')[0],
                    "shape": "ellipse"
            }
        )
        __nodes__[self.script] = node_diagram_id
        diagram["counter"] += 1
        
        # Add the script artifact
        node_diagram_id_script = str(diagram["counter"])
        g.add_node(node_diagram_id_script, 
            {
                    "text": self.script ,
                    "shape": "box"
            }
        )
        diagram["counter"] += 1
        g.add_edge(node_diagram_id_script, node_diagram_id)

        for to_node, loc in to_list:
            g.add_edge(node_diagram_id, to_node)
        
        if self.in_artifacts:
            for artifact in self.in_artifacts:
                if artifact.getLocation() in __nodes__:
                    g.add_edge(__nodes__[artifact.getLocation()], node_diagram_id)
                else:
                    from_nodes = artifact.parent.__graphWalk__(diagram)
                    for from_node, loc in from_nodes:
                        if loc in [art.getLocation() for art in self.in_artifacts]:
                            g.add_edge(from_node, node_diagram_id)
        
        return to_list


__visited__ = []
__samples__ = []
__nodes__ = {}
__gc__ = None
__jarvisFile__ = 'driver.py'
__sample_interm_files__ = set([])