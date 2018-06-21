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


def commit(xp_state : State):
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
                node = xp_state.gc.get_node(sourceKey)
                nodeid = node.get_id()
                n = xp_state.gc.create_node_version(nodeid)
            else:
                assert len(n) == 1
                return xp_state.gc.get_node_version(n[0])
        except:
            n = xp_state.gc.create_node_version(xp_state.gc.get_node(sourceKey).get_id())

        return n

    def stringify(v):
         # https://stackoverflow.com/a/22505259/9420936
        return hashlib.md5(json.dumps(str(v) , sort_keys=True).encode('utf-8')).hexdigest()

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
    elif type(latest_experiment_node_versions) == type([]) and len(latest_experiment_node_versions) > 0:
        # This code makes GRIT compatible with GroundTable
        try:
            [int(i) for i in latest_experiment_node_versions]
        except:
            latest_experiment_node_versions = [i.get_id() for i in latest_experiment_node_versions]
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
                'value' : 'Post', #change to 'Post' after exec
                'type' : 'STRING',
            }
    }, parent_ids=latest_experiment_node_versions)

    starts: Set[Union[Artifact, Literal]] = xp_state.eg.starts
    for node in starts:
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
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


def peek(xp_state : State, loc):

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

    def find_outputs(end):
        to_list = []
        for child in end.out_artifacts:
            to_list.append(child)
        return to_list

    # Begin
    sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    latest_experiment_node_versions = xp_state.gc.get_node_latest_versions(sourcekeySpec)
    if latest_experiment_node_versions == []:
        latest_experiment_node_versions = None
    assert latest_experiment_node_versions is None or len(latest_experiment_node_versions) == 1

    # Create new spec node that results from peek.
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
                'value' : "0",
                'type' : 'STRING',
            },
        'prepostExec':
            {
                'key' : 'prepostExec',
                'value' : 'Post', #change to 'Post' after exec
                'type' : 'STRING',
            }
    }, parent_ids=latest_experiment_node_versions)

    # Get output artifacts
    arts = xp_state.eg.d.keys() - xp_state.eg.starts
    for each in arts:
        if type(each) == Artifact:
            if each.loc == loc:
                outputs = find_outputs(each.parent)

    #creates a dummy node
    peekSpec = sourcekeySpec + '.' + specnode.get_name()
    dummykey = peekSpec + '.dummy'
    dummynode = safeCreateGetNode(dummykey, dummykey)
    dummynodev = safeCreateGetNodeVersion(dummykey)

    # Note: we do not need to necessarily create a new node for the model.pkl output

    # Initialize sets
    starts: Set[Union[Artifact, Literal]] = xp_state.eg.starts
    ghosts = {}
    literalsOrder = []

    # Create literal nodes and bindings for initial artifacts/literals.
    for node in starts:
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
            literalsOrder.append(sourcekeyLit)
            litnode = safeCreateGetNode(sourcekeyLit, sourcekeyLit)
            e1 = safeCreateGetEdge(sourcekeyLit, "null", specnode.get_id(), litnode.get_id())

            litnodev = xp_state.gc.create_node_version(litnode.get_id())
            print(sourcekeyLit)
            xp_state.gc.create_edge_version(e1.get_id(), specnodev.get_id(), litnodev.get_id())

            # Create binding nodes and edges to dummy node
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
                    bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                    e3 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
                    startslineage = safeCreateLineage(sourcekeyLit, 'null')
                    xp_state.gc.create_lineage_edge_version(startslineage.get_id(), bindnodev.get_id(), dummynodev.get_id())

                    # Bindings are singleton node versions
                    # Facilitates backward lookup (All trials with alpha=0.0)
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
                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                e4 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
                startslineage = safeCreateLineage(dummykey + '.edge.' + str(node.v), 'null')
                xp_state.gc.create_lineage_edge_version(startslineage.get_id(), bindnodev.get_id(), dummynodev.get_id())

                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                ghosts[bindnodev.get_id()] = (bindnode.get_name(), str(v))
                xp_state.gc.create_edge_version(e4.get_id(), litnodev.get_id(), bindnodev.get_id())

        elif type(node) == Artifact:
            sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(node.loc)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            artnodev = safeCreateGetNodeVersion(sourcekeyArt)
            e2 = safeCreateGetEdge(sourcekeyArt, "null", specnode.get_id(), artnode.get_id())
            startslineage = safeCreateLineage(dummykey + '.edge.art.' + 'null')
            xp_state.gc.create_lineage_edge_version(startslineage.get_id(), artnodev.get_id(), dummynodev.get_id())

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

# Iterate through output artifacts to link them
    for each in arts:
        if type(each) == Action:
            #make a dummy node version
            #print(each.funcName)
            dummyversion = xp_state.gc.create_node_version(dummynode.get_id())
            actionkey = sourcekeySpec + "." + each.funcName
            #print(each.in_artifacts)
            #print(each.out_artifacts)
            #print("out")
            for ins in each.in_artifacts:
                #print(ins)
                if type(ins) == Literal:
                    sourcekeyLit = sourcekeySpec + '.literal.' + ins.name
                    #print(sourcekeyLit)
                    litnode = safeCreateGetNode(sourcekeyLit, sourcekeyLit)
                    litnodev = safeCreateGetNodeVersion(sourcekeyLit)
                    inkey = actionkey + '.literal.in.' + ins.name
                    dummylineage = safeCreateLineage(inkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), litnodev.get_id(), dummyversion.get_id())

                if type(ins) == Artifact:
                    sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(node.loc)
                    #print(sourcekeyArt)
                    artnode = safeCreateGetNode(sourcekeyArt, "null")
                    inkey = actionkey + ins.loc
                    dummylineage = safeCreateLineage(inkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), artnodev.get_id(), dummyversion.get_id())

            #print("out")
            for outs in each.out_artifacts:
                #print(outs)
                if type(outs) == Literal:
                    sourcekeyLit = sourcekeySpec + '.literal.' + outs.name
                    #print(sourcekeyLit)
                    litnode = safeCreateGetNode(sourcekeyLit, sourcekeyLit)
                    litnodev = safeCreateGetNodeVersion(sourcekeyLit)
                    outkey = actionkey + '.literal.out.' + outs.name
                    dummylineage = safeCreateLineage(outkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), dummyversion.get_id(),litnodev.get_id())

                if type(outs) == Artifact:
                    sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(outs.loc)
                    #print(sourcekeyArt)
                    artnode = safeCreateGetNode(sourcekeyArt, "null")
                    artnodev = safeCreateGetNodeVersion(sourcekeyArt)
                    outkey = actionkey + '.artifact.out.' + stringify(outs.loc)
                    dummylineage = safeCreateLineage(outkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), dummyversion.get_id(), artnodev.get_id())

    # Switch to versioning directory
    original = os.getcwd()
    os.chdir(xp_state.versioningDirectory + '/' + xp_state.EXPERIMENT_NAME)

    #FIXME: parent_ids is wrong, should be list of ids
    # Creates a new node and version representing peek
    peekKey = peekSpec + '.peek'
    peekNode = safeCreateGetNode(peekKey, peekKey)
    peekNodev = xp_state.gc.create_node_version(peekNode.get_id(), tags = {
        'timestamp': {
            'key' : 'timestamp',
            'value' : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'type' : 'STRING'
        }
    }, parent_ids = specnode.get_name()) #does this need to be a list of parent ids? does this need to exist?

    # Create edge and version for peek
    peekEdge = safeCreateGetEdge(peekKey, 'null', specnode.get_id(), peekNode.get_id())
    xp_state.gc.create_edge_version(peekEdge.get_id(), specnodev.get_id(), peekNodev.get_id())

    # Creates a new node representing trials.
    trialkey = peekKey + '.trials'
    trialnode = safeCreateGetNode(trialkey, trialkey)
    trialEdge = safeCreateGetEdge(trialkey, 'null', peekNodev.get_id(), trialnode.get_id())
    lineage = safeCreateLineage(trialkey, 'null')

    # Go into trial directory
    os.chdir("0")

    # FIXME: parent_ids is wrong, should be list of ids
    # Creating a trial node version for the peeked trial
    trialnodev = xp_state.gc.create_node_version(trialnode.get_id(), tags = {
        'trial': {
            'key': 'trialnumber',
            'value' : "0",
            'type' : 'STRING'
        }
    }, parent_ids = peekNode.get_name())

    # Link single trial to starting artifacts
    # Linking all starts nodes to the trial node
    for s in starts:
        if type(s) == Artifact:
            sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(s.loc)
            #print(sourcekeyArt)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            lineageart = safeCreateLineage(trialkey + ".artifact." + stringify(s.loc))
            xp_state.gc.create_lineage_edge_version(lineageart.get_id(), trialnodev.get_id(), artnode.get_id())

    # link trial to output node
    for out in outputs:
        sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
        sourcekey = sourcekeySpec + '.artifact.' + stringify(out.loc)
        #print("outs")
        #print(sourcekey)
        outnode = safeCreateGetNode(sourcekey, sourcekey)
        outputnodev = xp_state.gc.create_node_version(outnode.get_id(), tags = {
            'value' : {
                'key' : 'output',
                'value' : out.loc,
                'type' : 'STRING'
            }
        })

        # Create lineage for the only trial peeked.
        lineagetrial = safeCreateLineage(trialkey + '.0' + out.loc, 'null')

        #print("lineage trial")
        #print(lineagetrial)
        #print("trial node")
        #print(trialnodev)

        xp_state.gc.create_lineage_edge_version(lineagetrial.get_id(), trialnodev.get_id(), outputnodev.get_id()) #Fix this

    # Go through the pkl files in directory
    files = [x for x in os.listdir('.')]
    #print("files {}".format(files))

    # Get the value of the output of the trial
    num_ = 0
    file = 'ghost_literal_' + str(num_) + '.pkl'
    while file in files:
        with open(file, 'rb') as f:
            value = dill.load(f)
            #print("value {}".format(value))
            files.remove(file)

        flag = False
        for num in range(len(literalsOrder)):
            for g in ghosts:
                if ghosts[g] == (literalsOrder[num], str(value)):
                    #print("GHOST")
                    #print(ghosts[g])
                    lineagetrial = safeCreateLineage(trialkey + '.lit.' + str(ghosts[g][1]), 'null')
                    xp_state.gc.create_lineage_edge_version(lineagetrial.get_id(), trialnodev.get_id(), g)
                    flag = True
                    break
            if flag:
                break
        num_ += 1
        file = 'ghost_literal_' + str(num_) + '.pkl'
    #print("FILE {}".format(file))

    os.chdir(original)


def fork(xp_state : State, inputCH):
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
        with open('experiment_graph.pkl', 'rb') as f:
            experimentg = dill.load(f)
        util.runProc('git checkout master')
        os.chdir(original)
        return experimentg

    sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    #gives you a list of most recent experiment versions
    latest_experiment_node_versions = xp_state.gc.get_node_latest_versions(sourcekeySpec)
    if latest_experiment_node_versions == []:
        latest_experiment_node_versions = None

    timestamps = [xp_state.gc.get_node_version(x).get_tags()['timestamp']['value'] for x in latest_experiment_node_versions]
    latest_experiment_nodev = latest_experiment_node_versions[timestamps.index(min(timestamps))]
    #you are at the latest_experiment_node

    forkedNodev = None

    history = xp_state.gc.get_node_history(sourcekeySpec)
    for each in history.keys():
        tags = xp_state.gc.get_node_version(history[each]).get_tags()
        if 'commitHash' in tags.keys():
            if tags['commitHash']['value'] == inputCH:
                forkedNodev = history[each]
                break;


    if forkedNodev is None:
        raise Exception("Cannot fork to node that does not exist.")
    if xp_state.gc.get_node_version(forkedNodev).get_tags()['prepostExec']['value'] == 'Post':
        raise Exception("Connot fork from a Post-Execution State.")

    # FIXME: parent_ids is wrong, should be list of ids
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
                'value' : 'Pre',
                'type' : 'STRING',
            }
    }, parent_ids=forkedNodev) #changed this from original

    #checkout previous version and nab experiment_graph.pkl
    experimentg = geteg(xp_state, inputCH)
    lineage = safeCreateLineage(sourcekeySpec, 'null')
    xp_state.gc.create_lineage_edge_version(lineage.get_id(), latest_experiment_nodev, forkedNodev)
    starts : Set[Union[Artifact, Literal]] = experimentg.starts
    for node in starts:
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
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


def pull(xp_state : State, loc):
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

    def find_outputs(end):
        to_list = []
        for child in end.out_artifacts:
            to_list.append(child)
        return to_list

    sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    latest_experiment_node_versions = xp_state.gc.get_node_latest_versions(sourcekeySpec)
    if latest_experiment_node_versions == []:
        latest_experiment_node_versions = None
    elif type(latest_experiment_node_versions) == type([]) and len(latest_experiment_node_versions) > 0:
        # This code makes GRIT compatible with GroundTable
        try:
            [int(i) for i in latest_experiment_node_versions]
        except:
            latest_experiment_node_versions = [i.get_id() for i in latest_experiment_node_versions]

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
        'sequenceNumber': 
            {
                'key' : 'sequenceNumber', 
                'value' : "0",
                'type' : 'STRING',
            },
        'prepostExec':
            {
                'key' : 'prepostExec',
                'value' : 'Post', #change to 'Post' after exec
                'type' : 'STRING',
            }
    }, parent_ids=latest_experiment_node_versions)

    arts = xp_state.eg.d.keys() - xp_state.eg.starts
    for each in arts:
        if type(each) == Artifact:
            if each.loc == loc:
                outputs = find_outputs(each.parent)
    # outputs is your list of final artifacts

    #creates a dummy node
    pullspec = sourcekeySpec + '.' + specnode.get_name()
    dummykey = pullspec + '.dummy'
    dummynode = safeCreateGetNode(dummykey, dummykey)
    dummynodev = safeCreateGetNodeVersion(dummykey)

    #links everything to the dummy node
    starts : Set[Union[Artifact, Literal]] = xp_state.eg.starts
    ghosts = {}
    literalsOrder = []
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
                                'type' : 'STRING'
                            }})
                    bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                    e3 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
                    startslineage = safeCreateLineage(sourcekeyLit, 'null')
                    xp_state.gc.create_lineage_edge_version(startslineage.get_id(), bindnodev.get_id(), dummynodev.get_id())
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
                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                e4 = safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
                startslineage = safeCreateLineage(sourcekeyBind, 'null')
                xp_state.gc.create_lineage_edge_version(startslineage.get_id(), bindnodev.get_id(), dummynodev.get_id())
                # Bindings are singleton node versions

                bindnodev = safeCreateGetNodeVersion(sourcekeyBind)
                ghosts[bindnodev.get_id()] = (bindnode.get_name(), str(node.v))
                xp_state.gc.create_edge_version(e4.get_id(), litnodev.get_id(), bindnodev.get_id())

        elif type(node) == Artifact:
            sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(node.loc)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            artnodev = safeCreateGetNodeVersion(sourcekeyArt)
            e2 = safeCreateGetEdge(sourcekeyArt, "null", specnode.get_id(), artnode.get_id())
            startslineage = safeCreateLineage(sourcekeyArt, 'null')
            xp_state.gc.create_lineage_edge_version(startslineage.get_id(), artnodev.get_id(), dummynodev.get_id())
            
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
    #for each in arts:
        #version the dummy node and make a lineage edge from in artifacts to dummy node
        #all out artifacts will have a lineage edge from dummy node.
    #NOTE: there may be overlap. Items in the in-artifacts may be present in the starts set, which 
    # was already linked to the dummy node
    for each in arts:
        if type(each) == Action:
            #make a dummy node version
            dummyversion = xp_state.gc.create_node_version(dummynode.get_id())
            actionkey = sourcekeySpec + "." + each.funcName
            for ins in each.in_artifacts:
                if type(ins) == Literal:
                    sourcekeyLit = sourcekeySpec + '.literal.' + ins.name
                    litnode = safeCreateGetNode(sourcekeyLit, sourcekeyLit)
                    litnodev = safeCreateGetNodeVersion(sourcekeyLit)
                    inkey = actionkey + '.literal.in.' + ins.name
                    dummylineage = safeCreateLineage(inkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), litnodev.get_id(), dummyversion.get_id())
                if type(ins) == Artifact:
                    sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(ins.loc)
                    artnode = safeCreateGetNode(sourcekeyArt, "null")
                    artnodev = safeCreateGetNodeVersion(sourcekeyArt)
                    inkey = actionkey + '.artifact.in.' + stringify(ins.loc)
                    dummylineage = safeCreateLineage(inkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), artnodev.get_id(), dummyversion.get_id())

            print("out")
            for outs in each.out_artifacts:
                print(outs)
                if type(outs) == Literal:
                    sourcekeyLit = sourcekeySpec + '.literal.' + outs.name
                    litnode = safeCreateGetNode(sourcekeyLit, sourcekeyLit)
                    litnodev = safeCreateGetNodeVersion(sourcekeyLit)
                    outkey = actionkey + '.literal.out.' + outs.name
                    dummylineage = safeCreateLineage(outkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), dummyversion.get_id(), litnodev.get_id())
                if type(outs) == Artifact:
                    sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(outs.loc)
                    artnode = safeCreateGetNode(sourcekeyArt, "null")
                    artnodev = safeCreateGetNodeVersion(sourcekeyArt)
                    outkey = actionkey + '.artifact.out.' + stringify(outs.loc)
                    dummylineage = safeCreateLineage(outkey, 'null')
                    xp_state.gc.create_lineage_edge_version(dummylineage.get_id(), dummyversion.get_id(), artnodev.get_id())
                
    #switch to versioning directory
    original = os.getcwd()
    os.chdir(xp_state.versioningDirectory + '/' + xp_state.EXPERIMENT_NAME)

    #FIXME: parent_ids is wrong
    #creates a new node and version representing pull
    pullkey = pullspec + '.pull'
    pullnode = safeCreateGetNode(pullkey, pullkey)
    pullnodev = xp_state.gc.create_node_version(pullnode.get_id(), tags = {
        'timestamp': {
            'key' : 'timestamp',
            'value' : datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'type' : 'STRING'
        }
    }, parent_ids = specnode.get_name())

    pullEdge = safeCreateGetEdge(pullkey, 'null', specnode.get_id(), pullnode.get_id())
    xp_state.gc.create_edge_version(pullEdge.get_id(), specnodev.get_id(), pullnodev.get_id())

    #creates a new node representing trials. Does each trial have its own node, or is it node versions?
    trialkey = pullkey + '.trials'
    trialnode = safeCreateGetNode(trialkey, trialkey)
    trialEdge = safeCreateGetEdge(trialkey, 'null', pullnode.get_id(), trialnode.get_id())
    lineage = safeCreateLineage(trialkey, 'null')
    # xp_state.gc.create_lineage_edge_version(lineage.get_id(), modelnode.get_id(), trialnode.get_id())

    #created trial, now need to link each of the trials to the output

    #FIXME: parent_ids is wrong
    #iterates through all files in current directory 
    filetemp = 'ghost_literal_'
    for each in os.listdir("."):
        #ignore git file
        if each == '.git':
            continue
        os.chdir(each)
        #creating a trial node version for each trial
        trialnodev = xp_state.gc.create_node_version(trialnode.get_id(), tags = {
            'trial': {
                'key': 'trialnumber',
                'value' : each,
                'type' : 'STRING'
            }
        }, parent_ids = pullnode.get_name())

        output_nodes = []

        #link every trial to starting artifacts
        #Im not super sure about this part? Linking all starts nodes to the trial node
        for s in starts:
            if type(s) == Artifact:
                sourcekeyArt = sourcekeySpec + '.artifact.' + stringify(s.loc)
                artnode = safeCreateGetNode(sourcekeyArt, "null")
                artnodev = safeCreateGetNodeVersion(sourcekeyArt)
                lineageart = safeCreateLineage(trialkey + ".artifact." + stringify(s.loc), 'null')
                xp_state.gc.create_lineage_edge_version(lineageart.get_id(), trialnodev.get_id(), artnodev.get_id())

        #link trials to output node
        for out in outputs:
            sourcekeySpec = 'flor.' + xp_state.EXPERIMENT_NAME
            sourcekey = sourcekeySpec + '.artifact.' + stringify(out.loc)
            outnode = safeCreateGetNode(sourcekey, sourcekey)
            outputnodev = xp_state.gc.create_node_version(outnode.get_id(), tags = {
                'value' : {
                    'key' : 'output',
                    'value' : out.loc,
                    'type' : 'STRING'
                }
            })
            lineagetrial = safeCreateLineage(trialkey + '.' + each + "." + out.loc, 'null')
            xp_state.gc.create_lineage_edge_version(lineagetrial.get_id(), trialnodev.get_id(), outputnodev.get_id())

        files = [x for x in os.listdir('.')]
        num_ = 0
        file = 'ghost_literal_' + str(num_) + '.pkl'
        while file in files:
            with open(file, 'rb') as f:
                value = dill.load(f)
                files.remove(file)

            flag = False
            for num in range(len(literalsOrder)):
                for g in ghosts:
                    if ghosts[g] == (literalsOrder[num], str(value)):
                        print("GHOST")
                        print(ghosts[g])
                        lineagetrial = safeCreateLineage(trialkey + '.lit.' + str(ghosts[g][1]), 'null')
                        xp_state.gc.create_lineage_edge_version(lineagetrial.get_id(), trialnodev.get_id(), g)
                        flag = True
                        break
                if flag:
                    break
            num_ += 1
            file = 'ghost_literal_' + str(num_) + '.pkl'
                
        os.chdir('..')

    os.chdir(original)


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
                                         fromNv.get_id(), toNv.get_id())