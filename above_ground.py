#!/usr/bin/env python3

from typing import Set, Union
import hashlib
import json
import datetime
import dill

import flor.util as util
from flor.stateful import State
from flor.object_model import Artifact, Action, Literal
import os
import subprocess


def commit(xp_state : State, prepost='Post'):
    def safeCreateGetNode(sourceKey, name, tags=None):
        # Work around small bug in ground client
        try:
            n = xp_state.gc.get_node(sourceKey)
            if n is None:
                n = xp_state.gc.create_node(sourceKey, name, tags)
        except:
            n = xp_state.gc.create_node(sourceKey, name, tags)

        return n

    def safeCreateGetEdge(sourceKey, name, fromNodeId, toNodeId, tags=None):
        try:
            n = xp_state.gc.get_edge(sourceKey)
            if n is None:
                n = xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)
        except:
            n = xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)

        return n

    def safeCreateGetNodeVersion(sourceKey):
        # Good for singleton node versions
        try:
            n = xp_state.gc.get_node_latest_versions(sourceKey)
            if n is None or n == []:
                n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())
            else:
                assert len(n) == 1
                return xp_state.gc.get_node_version(n[0])
        except:
            n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())

        return n

    def stringify(v):
         # https://stackoverflow.com/a/22505259/9420936
        return hashlib.md5(json.dumps(str(v) , sort_keys=True).encode('utf-8')).hexdigest()

    # def get_sha(versioningDirectory):
    #     sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=versioningDirectory).decode('ascii').strip()
    #     return sha

    def get_sha(directory):
        #FIXME: output contains the correct thing, but there is no version directory yet...
        original = os.getcwd()
        os.chdir(directory)
        output = subprocess.check_output('git log -1 --format=format:%H'.split()).decode()
        os.chdir(original)
        return output

    # Begin
    sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    latest_experiment_node_versions = xp_state.gc.get_node_latest_versions(sourcekeySpec)
    if latest_experiment_node_versions == []:
        latest_experiment_node_versions = None
    assert latest_experiment_node_versions is None or len(latest_experiment_node_versions) == 1

    # How does fork affect latest_experiment_node_versions?
        # Don't worry about it: managed by fork
        # Relying on valid pre-condition, we can always just get the latest node version

    specnodev = xp_state.gc.create_node_version(specnode.get_id(), tags={
        'timestamp':
            {
                'key' : 'timestamp',
                'value' : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'type' : 'STRING'
            },
        'commitHash':
            {
                'key' : 'commitHash',
                'value' : get_sha(xp_state.versioningDirectory + '/' + xp_state.EXPERIMENT_NAME),
                'type' : 'STRING',
            },
        'sequenceNumber': #potentially unneeded...can't find a good way to get sequence number
            {
                'key' : 'sequenceNumber', 
                'value' : "0", #fixme given a commit hash we'll have to search through for existing CH
                'type' : 'STRING',
            },
        'prepostExec':
            {
                'key' : 'prepostExec',
                'value' : prepost, #change to 'Post' after exec
                'type' : 'STRING',
            }
    }, parent_ids=latest_experiment_node_versions)

    starts : Set[Union[Artifact, Literal]] = xp_state.eg.starts
    for node in starts:
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
            litnode = safeCreateGetNode(sourcekeyLit, "null")
            e1 = safeCreateGetEdge(sourcekeyLit, "null", specnode.get_id(), litnode.get_id())

            litnodev = xp_state.gc.create_node_version(litnode.get_id())
            xp_state.gc.create_edge_version(e1.get_id(), specnodev.get_id(), litnodev.get_id())

            if node.__oneByOne__:
                for i, v in enumerate(node.v):
                    sourcekeyBind = sourcekeyLit + '.' + stringify(v)
                    bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                        'value':
                            {
                                'key': 'value',
                                'value': str(v),
                                'type' : 'STRING'
                            }})
                    e3 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())

                    # Bindings are singleton node versions
                    #   Facilitates backward lookup (All trials with alpha=0.0)

                    bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                    xp_state.gc.create_edge_version(e3.get_id(), litnodev.get_id(), bindnodev.get_id())
            else:
                sourcekeyBind = sourcekeyLit + '.' + stringify(node.v)
                bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                    'value':
                        {
                            'key': 'value',
                            'value': str(node.v),
                            'type': 'STRING'
                        }})
                e4 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())

                # Bindings are singleton node versions

                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                xp_state.gc.create_edge_version(e4.get_id(), litnodev.get_id(), bindnodev.get_id())

        elif type(node) == Artifact:
            sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(node.loc)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            e2 = safeCreateGetEdge(sourcekeyArt, "null", specnode.get_id(), artnode.get_id())

            # TODO: Get parent Verion of Spec, forward traverse to artifact versions. Find artifact version that is parent.

            artnodev = xp_state.gc.create_node_version(artnode.get_id(), tags={
                'checksum': {
                    'key': 'checksum',
                    'value': util.md5(node.loc),
                    'type': 'STRING'
                }
            })
            xp_state.gc.create_edge_version(e2.get_id(), specnodev.get_id(), artnodev.get_id())

        else:
            raise TypeError(
                "Action cannot be in set: starts")


def peek(xp_state : State):
    def safeCreateGetNode(sourceKey, name, tags=None):
        # Work around small bug in ground client
        try:
            n = xp_state.gc.get_node(sourceKey)
            if n is None:
                n = xp_state.gc.create_node(sourceKey, name, tags)
        except:
            n = xp_state.gc.create_node(sourceKey, name, tags)

        return n

    def safeCreateGetEdge(sourceKey, name, fromNodeId, toNodeId, tags=None):
        try:
            n = xp_state.gc.get_edge(sourceKey)
            if n is None:
                n = xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)
        except:
            n = xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)

        return n

    def safeCreateGetNodeVersion(sourceKey):
        # Good for singleton node versions
        try:
            n = xp_state.gc.get_node_latest_versions(sourceKey)
            if n is None or n == []:
                n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())
            else:
                assert len(n) == 1
                return xp_state.gc.get_node_version(n[0])
        except:
            n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())

        return n

    def safeCreateLineage(sourceKey, name, tags=None):
        print(sourceKey)
        try:
            n = xp_state.gc.get_lineage_edge(sourceKey)
            if n is None or n == []:
                n = xp_state.gc.create_lineage_edge(sourceKey, name, tags)
        except:
            n = xp_state.gc.create_lineage_edge(sourceKey, name, tags)
        return n

    def stringify(v):
         # https://stackoverflow.com/a/22505259/9420936
        return hashlib.md5(json.dumps(str(v) , sort_keys=True).encode('utf-8')).hexdigest()

    def get_sha(directory):
        original = os.getcwd()
        os.chdir(directory)
        output = subprocess.check_output('git log -1 --format=format:%H'.split()).decode()
        os.chdir(original)
        return output


    # Begin
    sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    latest_experiment_node_versions = xp_state.gc.get_node_latest_versions(sourcekeySpec)
    if latest_experiment_node_versions == []:
        latest_experiment_node_versions = None
    assert latest_experiment_node_versions is None or len(latest_experiment_node_versions) == 1

    specnodev = xp_state.gc.create_node_version(specnode.get_id(), tags={
        'timestamp':
            {
                'key' : 'timestamp',
                'value' : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'type' : 'STRING'
            },
        'commitHash':
            {
                'key' : 'commitHash',
                'value' : get_sha(xp_state.versioningDirectory + '/' + xp_state.EXPERIMENT_NAME),
                'type' : 'STRING',
            },
        'sequenceNumber': #potentially unneeded...can't find a good way to get sequence number
            {
                'key' : 'sequenceNumber',
                'value' : "0", #fixme given a commit hash we'll have to search through for existing CH
                'type' : 'STRING',
            },
        'prepostExec':
            {
                'key' : 'prepostExec',
                'value' : prepost, #change to 'Post' after exec
                'type' : 'STRING',
            }
    }, parent_ids=latest_experiment_node_versions)

    #creates a dummy node
    peekSpec = sourcekeySpec + '.' + specnode.get_name()
    dummykey = peekSpec + '.dummy'
    dummynode = safeCreateGetNode(dummykey, dummykey)

    #creates a new node for the model.pkl
    modelkey = peekSpec + '.model'
    modelnode = safeCreateGetNode(modelkey, modelkey)

    # Connect lineage edge between dummy and model
    lineage = safeCreateLineage(peekSpec, 'null')
    xp_state.gc.create_lineage_edge_version(lineage.get_id(), dummynode.get_id(), modelnode.get_id())


    starts: Set[Union[Artifact, Literal]] = xp_state.eg.starts
    ghosts = {}
    literalsOrder = []

    # Create literal nodes and bindings
    for node in starts:
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
            literalsOrder.append(sourcekeyLit)
            litnode = safeCreateGetNode(sourcekeyLit, sourcekeyLit)
            e1 = safeCreateGetEdge(sourcekeyLit, "null", specnode.get_id(), litnode.get_id())

            litnodev = xp_state.gc.create_node_version(litnode.get_id())
            xp_state.gc.create_edge_version(e1.get_id(), specnodev.get_id(), litnodev.get_id())

            if node.__oneByOne__:
                for i, v in enumerate(node.v):
                    sourcekeyBind = sourcekeyLit + '.' + stringify(v)
                    bindnode = safeCreateGetNode(sourcekeyBind, sourcekeyLit, tags={
                        'value':
                            {
                                'key': 'value',
                                'value': str(v),
                                'type': 'STRING'
                            }})
                    e3 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
                    startslineage = safeCreateLineage(dummykey + '.edge.' + str(v), 'null')
                    xp_state.gc.create_lineage_edge_version(startslineage.get_id(), bindnode.get_id(),
                                                            dummynode.get_id())
                    # Bindings are singleton node versions
                    #   Facilitates backward lookup (All trials with alpha=0.0)
                    bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                    ghosts[bindnodev.get_id()] = (bindnode.get_name(), str(v))
                    xp_state.gc.create_edge_version(e3.get_id(), litnodev.get_id(), bindnodev.get_id())
            else:
                sourcekeyBind = sourcekeyLit + '.' + stringify(node.v)
                bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                    'value':
                        {
                            'key': 'value',
                            'value': str(node.v),
                            'type': 'STRING'
                        }})
                e4 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
                startslineage = safeCreateLineage(dummykey + '.edge.' + str(node.v), 'null')
                xp_state.gc.create_lineage_edge_version(startslineage.get_id(), bindnode.get_id(), dummynode.get_id())
                # Bindings are singleton node versions

                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                ghost[bindnodev] = (bindnode.get_name(), str(v))
                xp_state.gc.create_edge_version(e4.get_id(), litnodev.get_id(), bindnodev.get_id())

        elif type(node) == Artifact:
            sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(node.loc)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            e2 = safeCreateGetEdge(sourcekeyArt, "null", specnode.get_id(), artnode.get_id())
            startslineage = safeCreateLineage(dummykey + '.edge.art.' +, 'null')
            xp_state.gc.create_lineage_edge_version(startslineage.get_id(), artnode.get_id(), dummynode.get_id())

            artnodev = xp_state.gc.create_node_version(artnode.get_id(), tags={
                'checksum': {
                    'key': 'checksum',
                    'value': util.md5(node.loc),
                    'type': 'STRING'
                }
            })
            xp_state.gc.create_edge_version(e2.get_id(), specnodev.get_id(), artnodev.get_id())
        else:
            raise TypeError(
                "Action cannot be in set: starts")


    #switch to versioning directory
    original_dir = os.getcwd()
    os.chdir(xp_state.versioningDirectory + '/' + xp_state.EXPERIMENT_NAME)

    #creates a new node and version representing pull
    peekKey = 'flor.' + specnode.get_name() + '.pull'
    peekNode = safeCreateGetNode(peekKey, peekKey)
    peekNodev = xp_state.gc.create_node_version(peekNode.get_id(), tags = {
        'timestamp': {
            'key' : 'timestamp',
            'value' : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'type' : 'STRING'
        }
    }, parent_ids = specnode.get_name()); #does this need to be a list of parent ids? does this need to exist?

    pullEdge = safeCreateGetEdge(peekKey, 'null', specnode.get_id(), peekNode.get_id())
    xp_state.gc.create_edge_version(pullEdge.get_id(), specnodev.get_id(), peekNodev.get_id())

    #creates a new node representing trials.
    trialkey = 'flor.' + specnode.get_name() + '.trials'
    trialnode = safeCreateGetNode(trialkey, trialkey)
    trialEdge = safeCreateGetEdge(trialkey, 'null', peekNode.get_id(), trialnode.get_id())
    lineage = xp_state.gc.create_lineage_edge(sourcekeySpec, 'null')
    xp_state.gc.create_lineage_edge_version(lineage.get_id(), modelnode.get_id(), trialnode.get_id())



def fork(xp_state : State, inputCH):

    #FIXME: figure out alternative way to get hash (see gitlog in commit above)
    #verify the lineage edge code below

    def safeCreateGetNode(sourceKey, name, tags=None):
        # Work around small bug in ground client
        try:
            n = xp_state.gc.get_node(sourceKey)
            if n is None:
                n = xp_state.gc.create_node(sourceKey, name, tags)
        except:
            n = xp_state.gc.create_node(sourceKey, name, tags)

        return n

    def safeCreateGetEdge(sourceKey, name, fromNodeId, toNodeId, tags=None):
        try:
            n = xp_state.gc.get_edge(sourceKey)
            if n is None:
                n = xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)
        except:
            n = xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)

        return n

    def safeCreateGetNodeVersion(sourceKey):
        # Good for singleton node versions
        try:
            n = xp_state.gc.get_node_latest_versions(sourceKey)
            if n is None or n == []:
                n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())
            else:
                assert len(n) == 1
                return xp_state.gc.get_node_version(n[0])
        except:
            n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())

        return n

    def safeCreateLineage(sourceKey, name, tags=None):
        try:
            n = xp_state.gc.create_lineage_edge(sourceKey, name, tags)
        except:
            n = xp_state.gc.create_lineage_edge(sourceKey, name, tags)

        return n

    def stringify(v):
         # https://stackoverflow.com/a/22505259/9420936
        return hashlib.md5(json.dumps(str(v) , sort_keys=True).encode('utf-8')).hexdigest()

    def get_sha(directory):
        output = subprocess.check_output('git log -1 --format=format:%H'.split()).decode()


    def geteg(xp_state, inputCH):
        original = os.getcwd()
        os.chdir(xp_state.versioningDirectory + '/' + xp_state.EXPERIMENT_NAME)
        util.runProc('git checkout ' + inputCH)
        os.chdir("0/")
        with open('experiment_graph.pkl', 'rb') as f:
            experimentg = dill.load(f)
        # for each in experimentg.d.keys():
        #     temp = experimentg.d[each]
        #     for x in temp:
        #         if type(x) is not set and type(x) is not Action:
        #             print(x.getLocation())
        #             input()
        util.runProc('git checkout master')
        os.chdir(original)
        return experimentg

    sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    #gives you a list of most recent experiment versions
    latest_experiment_node_versions = xp_state.gc.get_node_latest_versions(sourcekeySpec)
    if latest_experiment_node_versions == []:
        latest_experiment_node_versions = None

    # print(sourcekeySpec)
    # print(xp_state.gc.get_node_history(sourcekeySpec))
    # print(xp_state.gc.get_node_latest_versions('flor.plate_demo'))
    # input()
    timestamps = [xp_state.gc.get_node_version(x).get_tags()['timestamp']['value'] for x in latest_experiment_node_versions]
    latest_experiment_nodev = latest_experiment_node_versions[timestamps.index(min(timestamps))]
    #you are at the latest_experiment_node

    forkedNodev = None
    # for each in latest_experiment_node_versions:
    #     if flag:
    #         break
    #     history = xp_state.gc.get_node_version_adjacent_lineage(each)
    #     for node in history:
    #         #does this return tags d for every node?
    #         #assume history returns a list of nodeIds
    #         d = xp_state.gc.getNodeVersion(node)
    #         if d['commitHash'] == inputCH and d['prepostExec']:
    #             forkedNodev = node
    #             flag = True

    history = xp_state.gc.get_node_history(sourcekeySpec)
    for each in history.keys():
        tags = xp_state.gc.get_node_version(history[each]).get_tags()
        print(tags)
        if 'commitHash' in tags.keys():
            if tags['commitHash']['value'] == inputCH:
                forkedNodev = history[each]
                break;


    #TODO: fix all the stuff below, which contains a lot of speculation
    #get specnodev corresponding to forkedNode?
    if forkedNodev is None:
        raise Exception("Cannot fork to node that does not exist.")
    # How does fork affect latest_experiment_node_versions?
        # Don't worry about it: managed by fork
        # Relying on valid pre-condition, we can always just get the latest node version
    if xp_state.gc.get_node_version(forkedNodev).get_tags()['prepostExec']['value'] == 'Post':
        raise Exception("Connot fork from a Post-Execution State.")

    specnodev = xp_state.gc.create_node_version(specnode.get_id(), tags={
        'timestamp':
            {
                'key' : 'timestamp',
                'value' : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'type' : 'STRING'
            },
        'commitHash':
            {
                'key' : 'commitHash',
                'value' : inputCH,
                'type' : 'STRING',
            },
        'sequenceNumber':
            {
                'key' : 'sequenceNumber', #useless currently
                'value' : str(int(xp_state.gc.get_node_version(forkedNodev).get_tags()['sequenceNumber']['value']) + 1),
                'type' : 'STRING',
            },
        'prepostExec':
            {
                'key' : 'prepostExec',
                'value' : 'Pre', #can only fork from pre state
                'type' : 'STRING',
            }
    }, parent_ids=forkedNodev) #changed this from original

    #TODO: increment seq #, add lineage edges to everything
    #TODO: specify lineage edges?

    #checkout previous version and nab experiment_graph.pkl
    experimentg = geteg(xp_state, inputCH)

    #TODO: lineage is returning None. Check what ground does and if its erroring out and returning None
    #make sure it is node, not node version
    lineage = safeCreateLineage(sourcekeySpec, 'null')
    #i think we use version id
    #what is rich version id?
    xp_state.gc.create_lineage_edge_version(lineage.get_id(), latest_experiment_nodev, forkedNodev)
    starts : Set[Union[Artifact, Literal]] = experimentg.starts
    # print(starts)
    # print(specnodev)
     #lineage is none right now
    # print(lineage)
    # input()

    print(starts)
    for node in starts:
        # input()
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
            litnode = safeCreateGetNode(sourcekeyLit, "null")
            e1 = safeCreateGetEdge(sourcekeyLit, "null", specnode.get_id(), litnode.get_id())

            litnodev = xp_state.gc.create_node_version(litnode.get_id())
            xp_state.gc.create_edge_version(e1.get_id(), specnodev.get_id(), litnodev.get_id())

            if node.__oneByOne__:
                for i, v in enumerate(node.v):
                    sourcekeyBind = sourcekeyLit + '.' + stringify(v)
                    bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                        'value':
                            {
                                'key': 'value',
                                'value': str(v),
                                'type' : 'STRING'
                            }})
                    e3 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())

                    # Bindings are singleton node versions
                    #   Facilitates backward lookup (All trials with alpha=0.0)

                    bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                    xp_state.gc.create_edge_version(e3.get_id(), litnodev.get_id(), bindnodev.get_id())
            else:
                sourcekeyBind = sourcekeyLit + '.' + stringify(node.v)
                bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                    'value':
                        {
                            'key': 'value',
                            'value': str(node.v),
                            'type': 'STRING'
                        }})
                e4 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())

                # Bindings are singleton node versions

                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                xp_state.gc.create_edge_version(e4.get_id(), litnodev.get_id(), bindnodev.get_id())

        elif type(node) == Artifact:
            sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(node.loc)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            e2 = safeCreateGetEdge(sourcekeyArt, "null", specnode.get_id(), artnode.get_id())

            # TODO: Get parent Verion of Spec, forward traverse to artifact versions. Find artifact version that is parent.
            # TODO: node.loc here is tweets.csv....why is tweets showing up here?
            artnodev = xp_state.gc.create_node_version(artnode.get_id(), tags={
                'checksum': {
                    'key': 'checksum',
                    'value': util.md5(node.loc),
                    'type': 'STRING'
                }
            })
            xp_state.gc.create_edge_version(e2.get_id(), specnodev.get_id(), artnodev.get_id())

        else:
            raise TypeError(
                "Action cannot be in set: starts")


def __tags_equal__(groundtag, mytag):
    groundtagprime = {}
    for kee in groundtag:
        groundtagprime[kee] = {}
        for kii in groundtag[kee]:
            if kii != 'id':
                groundtagprime[kee][kii] = groundtag[kee][kii]
    return groundtagprime == mytag

def newExperimentVersion(xp_state: State):
    # -- caution with fixed values like 'florExperiment', allowing for early Ground Ref prototype

    # The name of this experiment is in a tag in the nodeVersion of 'florExperiment'
    latest_experiment_node_versions = [x for x in xp_state.gc.getNodeLatestVersions('florExperiment')
                                       if xp_state.gc.getNodeVersion(x).get_tags()['experimentName'][
                                           'value'] == xp_state.EXPERIMENT_NAME]

    # This experiment may have previous versions, then the most recents are the parents
    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('florExperiment').get_id(),
                                       tags={
                                           'experimentName': {
                                               'key': 'experimentName',
                                               'value': xp_state.EXPERIMENT_NAME,
                                               'type': 'STRING'
                                           }},
                                       parentIds=latest_experiment_node_versions)

def newTrialVersion(xp_state : State, literals, artifacts):

    my_tag = {}
    for i, kee in enumerate(literals):
        my_tag['literalName' + str(i)] = {
            'key' : 'literalName' + str(i),
            'value': kee,
            'type': 'STRING'
        }
        my_tag['literalValue' + str(i)] = {
            'key' : 'literalValue' + str(i),
            'value' : str(literals[kee]),
            'type' : 'STRING'
        }
    for i, kee in enumerate(artifacts):
        my_tag['artifactName' + str(i)] = {
            'key' : 'artifactName' + str(i),
            'value': kee,
            'type': 'STRING'
        }
        my_tag['artifactMD5_' + str(i)] = {
            'key' : 'artifactMD5_' + str(i),
            'value' : artifacts[kee],
            'type' : 'STRING'
        }

    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('florTrial').get_id(),
                                         tags=my_tag)

def newLiteralVersion(xp_state : State , literalName, literalValue):

    my_tag = {     'literalName' : {
                         'key' : 'literalName',
                         'value' : literalName,
                         'type' : 'STRING'
                     },
                     'literalValue' : {
                         'key': 'literalValue',
                         'value' : str(literalValue),
                         'type' : 'STRING'
                     }
                 }

    candidate_nvs = [xp_state.gc.getNodeVersion(str(x)) for x in xp_state.gc.getNodeLatestVersions('florLiteral')
                     if __tags_equal__(xp_state.gc.getNodeVersion(str(x)).get_tags(), my_tag)]
    assert len(candidate_nvs) <= 1

    if len(candidate_nvs) == 1:
        return candidate_nvs[0]
    else:
        return xp_state.gc.createNodeVersion(xp_state.gc.getNode('florLiteral').get_id(),
                                         tags = my_tag)

def newArtifactVersion(xp_state : State, artifactName):
    # Connect artifact versions to parents offline
    # What's important at this level is the tags
    # What artifact meta-data do we care about

    my_tag = {
                   'artifactName': {
                       'key': 'artifactName',
                       'value': artifactName,
                       'type': 'STRING'
                   }
               }

    candidate_nvs = [xp_state.gc.getNodeVersion(str(x)) for x in xp_state.gc.getNodeLatestVersions('florArtifact')
                     if __tags_equal__(xp_state.gc.getNodeVersion(str(x)).get_tags(), my_tag)]
    assert len(candidate_nvs) <= 1

    if len(candidate_nvs) == 1:
        return candidate_nvs[0]
    else:
        return xp_state.gc.createNodeVersion(xp_state.gc.getNode('florArtifact').get_id(),
                                       tags=my_tag)

def newActionVersion(xp_state : State, actionName):
    my_tag = {     'actionName' : {
                         'key' : 'actionName',
                         'value' : actionName,
                         'type' : 'STRING'
                 }
    }
    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('florAction').get_id(),
                                         tags=my_tag)


def newExperimentTrialEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'florExperimentflorTrial')

def newTrialLiteralEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'florTrialflorLiteral')

def newTrialArtifactEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'florTrialflorArtifact')

def newLiteralActionEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'florLiteralflorAction')

def newArtifactActionEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'florArtifactflorAction')

def newActionArtifactEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'florActionflorArtifact')

def __newEdgeVersion__(xp_state : State, fromNv, toNv, edgeKey):
    return xp_state.gc.createEdgeVersion(xp_state.gc.getEdge(edgeKey).get_id(),
                                         fromNv.get_id(),
toNv.get_id())