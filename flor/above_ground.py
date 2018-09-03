#!/usr/bin/env python3

from typing import Set, Union
import hashlib
import json
import datetime
import dill

import flor.util as util
from flor.stateful import State
from flor.object_model import Artifact, Action, Literal
from flor.experiment_graph import deserialize as eg_deserialize

import os
import subprocess


class ContextTracker(object):

    def __init__(self, xp_state: State):
        self.xp_state = xp_state
        self.sourcekeySpec = 'flor.' + self.xp_state.EXPERIMENT_NAME
        self.specnode = self.__safeCreateGetNode__(self.sourcekeySpec, "null")
        self.versioning_directory = os.path.join(self.xp_state.versioningDirectory, self.xp_state.EXPERIMENT_NAME)

    def __get_artifact_version_parent__(self, this_spec_version, artifact_loc):
        """
        WARNING: This method may need more work
        This method will play important role in Artifact version consolidation, backward provenance tracking
        :param this_spec_version:
        :return:
        """
        # WE DO NOT CURRENTLY SUPPORT MERGING (ONLY FORK) SO CAN HAVE AT MOST ONE PARENT
        parentIds = this_spec_version.get_parent_ids()
        if not parentIds:
            return None
        assert len(parentIds) == 1, "Too many candidate parents for the artifact version: {}".format(parentIds)
        parentSpecNodeVId = parentIds[0]
        result = self.xp_state.gc.get_node_version_adjacent_edge_versions(
            parentSpecNodeVId, self.sourcekeySpec + '.artifact.' + self.__stringify__(artifact_loc))
        result = result['outward']
        assert len(result) == 1, "A spec node version can't have many outward edge versions to a {} artifact version.".format(artifact_loc)
        result = result[0]
        result.get_to_node_version_start_id()
        return self.xp_state.gc.get_node_version(result.get_to_node_version_start_id())

    def __get_recent_specnodev__(self):
        latest_experiment_node_versions = self.xp_state.gc.get_node_latest_versions(self.sourcekeySpec)
        if latest_experiment_node_versions == []:
            latest_experiment_node_versions = None
        return latest_experiment_node_versions

    def __get_recent_specnodev_id__(self):
        latest_experiment_node_versions = self.xp_state.gc.get_node_latest_versions(self.sourcekeySpec)
        if latest_experiment_node_versions == []:
            latest_experiment_node_versions = None
        elif type(latest_experiment_node_versions) == type([]) and len(latest_experiment_node_versions) > 0:
            # This code makes GRIT compatible with GroundTable
            try:
                [int(i) for i in latest_experiment_node_versions]
            except:
                latest_experiment_node_versions = [i.get_id() for i in latest_experiment_node_versions]
        return latest_experiment_node_versions


    def __new_spec_nodev__(self):
        latest_experiment_node_versions = self.__get_recent_specnodev__()
        if latest_experiment_node_versions and len(latest_experiment_node_versions) > 0:
            maxtstamp = max([each.get_tags()['timestamp'].get_value() for each in latest_experiment_node_versions])
            latest_experiment_node_versions = list(filter(lambda x: x.get_tags()['timestamp'].get_value() == maxtstamp, latest_experiment_node_versions))
            assert len(latest_experiment_node_versions) == 1, "Error, multiple latest specnode versions have equal timestamps"
            parent_ids = [latest_experiment_node_versions[0].get_id()]
        else:
            parent_ids = None

        if parent_ids:
            parent = parent_ids[0]
            seq_num = str(int(parent.get_tags()['sequenceNumber'].get_value()) + 1)
        else:
            seq_num = "0"

        self.specnodev = self.xp_state.gc.create_node_version(self.specnode.get_id(), tags={
            'timestamp':
                {
                    'key': 'timestamp',
                    'value': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    'type': 'STRING'
                },
            'commitHash': # this commit; not parent commit
                {
                    'key': 'commitHash',
                    'value': self.__get_sha__(self.xp_state.versioningDirectory + '/' + self.xp_state.EXPERIMENT_NAME),
                    'type': 'STRING',
                },
            'sequenceNumber':  # potentially unneeded...can't find a good way to get sequence number
                {
                    'key': 'sequenceNumber',
                    'value': seq_num,
                    'type': 'STRING',
                }
        }, parent_ids=parent_ids)

    def __safeCreateGetNode__(self, sourceKey, name, tags=None):
        # Work around small bug in ground client
        try:
            n = self.xp_state.gc.get_node(sourceKey)
            if n is None:
                n = self.xp_state.gc.create_node(sourceKey, name, tags)
        except:
            n = self.xp_state.gc.create_node(sourceKey, name, tags)

        return n

    def __safeCreateGetEdge__(self, sourceKey, name, fromNodeId, toNodeId, tags=None):
        try:
            n = self.xp_state.gc.get_edge(sourceKey)
            if n is None:
                n = self.xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)
        except:
            n = self.xp_state.gc.create_edge(sourceKey, name, fromNodeId, toNodeId, tags)

        return n

    def __safeCreateGetNodeVersion__(self, sourceKey):
        # Good for singleton node versions
        try:
            n = self.xp_state.gc.get_node_latest_versions(sourceKey)
            if n is None or n == []:
                node = self.xp_state.gc.get_node(sourceKey)
                nodeid = node.get_id()
                n = self.xp_state.gc.create_node_version(nodeid)
            else:
                assert len(n) == 1
                return self.xp_state.gc.get_node_version(n[0])
        except:
            n = self.xp_state.gc.create_node_version(self.xp_state.gc.get_node(sourceKey).get_id())

        return n

    def __safeCreateLineage__(self, sourceKey, name, tags=None):
        try:
            n = self.xp_state.gc.get_lineage_edge(sourceKey)
            if n is None or n == []:
                n = self.xp_state.gc.create_lineage_edge(sourceKey, name, tags)
        except:
            n = self.xp_state.gc.create_lineage_edge(sourceKey, name, tags)
        return n

    @staticmethod
    def __stringify__(v):
         # https://stackoverflow.com/a/22505259/9420936
        return hashlib.md5(json.dumps(str(v) , sort_keys=True).encode('utf-8')).hexdigest()

    @staticmethod
    def __get_sha__(directory):
        # FIXME: output contains the correct thing, but there is no version directory yet...
        original = os.getcwd()
        os.chdir(directory)
        output = subprocess.check_output('git log -1 --format=format:%H'.split()).decode()
        os.chdir(original)
        return output
    @staticmethod
    def __find_outputs__(end):
        to_list = []
        for child in end.out_artifacts:
            to_list.append(child)
        return to_list

class CommitTracker(ContextTracker):

    def __init__(self, xp_state):
        super().__init__(xp_state)

    def commit(self):
        self.__new_spec_nodev__()

        starts: Set[Union[Artifact, Literal]] = self.xp_state.eg.starts
        for node in starts:
            if type(node) == Literal:
                sourcekeyLit = self.sourcekeySpec + '.literal.' + node.name
                litnode = self.__safeCreateGetNode__(sourcekeyLit, sourcekeyLit)
                e1 = self.__safeCreateGetEdge__(sourcekeyLit, "null", self.specnode.get_id(), litnode.get_id())

                # TODO: add Literal versioning here.
                # TODO: serialize literal for referencing. Change literal tag. Point to serialized file rather than save value.

                litnodev = self.xp_state.gc.create_node_version(litnode.get_id())
                self.xp_state.gc.create_edge_version(e1.get_id(), self.specnodev.get_id(), litnodev.get_id())

                if node.__oneByOne__:
                    for i, v in enumerate(node.v):
                        sourcekeyBind = sourcekeyLit + '.' + self.__stringify__(v)
                        bindnode = self.__safeCreateGetNode__(sourcekeyBind, "null", tags={
                            'value':
                                {
                                    'key': 'value',
                                    'value': str(v),
                                    'type': 'STRING'
                                }})
                        e3 = self.__safeCreateGetEdge__(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())

                        # Bindings are singleton node versions
                        #   Facilitates backward lookup (All trials with alpha=0.0)

                        bindnodev = self.__safeCreateGetNodeVersion__(sourcekeyBind)
                        self.xp_state.gc.create_edge_version(e3.get_id(), litnodev.get_id(), bindnodev.get_id())
                else:
                    sourcekeyBind = sourcekeyLit + '.' + self.__stringify__(node.v)
                    bindnode = self.__safeCreateGetNode__(sourcekeyBind, "null", tags={
                        'value':
                            {
                                'key': 'value',
                                'value': str(node.v),
                                'type': 'STRING'
                            }})
                    e4 = self.__safeCreateGetEdge__(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())

                    # Bindings are singleton node versions

                    bindnodev = self.__safeCreateGetNodeVersion__(sourcekeyBind)
                    self.xp_state.gc.create_edge_version(e4.get_id(), litnodev.get_id(), bindnodev.get_id())

            elif type(node) == Artifact:
                sourcekeyArt = self.sourcekeySpec + '.artifact.' + self.__stringify__(node.loc)
                artnode = self.__safeCreateGetNode__(sourcekeyArt, "null")
                e2 = self.__safeCreateGetEdge__(sourcekeyArt, "null", self.specnode.get_id(), artnode.get_id())

                # TODO: Get parent Verion of Spec, forward traverse to artifact versions. Find artifact version that is parent.
                # TODO: Extend support to non-local artifacts, e.g. s3

                artnodev = self.xp_state.gc.create_node_version(artnode.get_id(), tags={
                    'checksum': {
                        'key': 'checksum',
                        'value': util.md5(node.loc),
                        'type': 'STRING'
                    },
                    'location': {
                        'key': 'location',
                        'value': node.loc,
                        'type': 'STRING'
                    }
                })
                self.xp_state.gc.create_edge_version(e2.get_id(), self.specnodev.get_id(), artnodev.get_id())

            else:
                raise TypeError(
                    "Action cannot be in set: starts")

class PullTracker(ContextTracker):

    def __init__(self, xp_state):
        super().__init__(xp_state)

    def __get_num_trails__(self):
        return len([each for each in os.listdir(self.versioning_directory) if util.isNumber(each)])

    def pull(self, loc):
        arts = self.xp_state.eg.d.keys() - self.xp_state.eg.starts

        for art in arts:
            if type(art) == Artifact and art.loc == loc:
                outputs = self.__find_outputs__(art.parent)
        # Outputs is your list of final artifacts

        # PullNode
        pullspec = self.sourcekeySpec + '.pull'
        pullnode = self.__safeCreateGetNode__(pullspec, pullspec)

        # PullNode -> SpecNode
        e1 = self.__safeCreateGetEdge__(pullspec, "null", pullnode.get_id(), self.specnode.get_id())

        # Version: PullNodeV -> SpecNodeV
        specnodev = self.__get_recent_specnodev__()
        if specnodev and len(specnodev) > 1:
            maxtstamp = max([each.get_tags()['timestamp'].get_value() for each in specnodev])
            specnodev = list(filter(lambda x: x.get_tags()['timestamp'].get_value() == maxtstamp, specnodev))
        assert len(specnodev) == 1, "Error, multiple latest specnode versions have equal timestamps"
        specnodev = specnodev[0]
        specnodev_id = specnodev.get_id()
        pullnodev = self.xp_state.gc.create_node_version(pullnode.get_id(), tags = {
            'timestamp':
                {
                    'key': 'timestamp',
                    'value': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    'type': 'STRING'
                }
        })
        self.xp_state.gc.create_edge_version(e1.get_id(), pullnodev.get_id(), specnodev_id)

        # TrialNode
        trialspec = pullspec + '.trial'
        trialnode = self.__safeCreateGetNode__(trialspec, trialspec)

        # PullNode -> TrialNode
        e2 = self.__safeCreateGetEdge__(trialspec, "null", pullnode.get_id(), trialnode.get_id())

        # Version: PullNodeV {> TrialNodeV
        trial_node_versions = []
        for i in range(self.__get_num_trails__()):
            trialnodev = self.xp_state.gc.create_node_version(trialnode.get_id())
            trial_node_versions.append(trialnodev)
            self.xp_state.gc.create_edge_version(e2.get_id(), pullnodev.get_id(), trialnodev.get_id())

        # Prepare the necessary lineage edges
        starting_literal = self.__safeCreateLineage__("starting_literal", "starting_literal")

        # Lineage-Edge Link Trial to Elements in the Start set (black dashed-lines in whiteboard diagram)
        for i, trialnodev in enumerate(trial_node_versions):
            with util.chinto(os.path.join(self.versioning_directory, str(i))):
                eg = eg_deserialize()
                for start in eg.starts:
                    if type(start) == Literal:
                        # WRONG: Don't bind a trial version to every binding; only to ITS bindings!!
                        if start.__oneByOne__:
                            for j, v in enumerate(start.v):
                                sourcekeybind = self.sourcekeySpec + '.literal.' + start.name + \
                                            '.' + self.__stringify__(v)
                                lvs = self.xp_state.gc.get_node_latest_versions(sourcekeybind)
                                assert len(lvs) == 1
                                lv = lvs[0]
                                self.xp_state.gc.create_lineage_edge_version(starting_literal.get_id(), lv.get_id(),
                                                                             trialnodev.get_id())
                        else:
                            sourcekeybind = self.sourcekeySpec + '.literal.' + start.name + \
                                            '.' + self.__stringify__(start.v)
                            lvs = self.xp_state.gc.get_node_latest_versions(sourcekeybind)
                            assert len(lvs) == 1
                            lv = lvs[0]
                            # GRIT COMPATIBLE
                            self.xp_state.gc.create_lineage_edge_version(starting_literal.get_id(), lv.get_id(),
                                                                         trialnodev.get_id())
                    else:
                        # NOTHING TO DO, we know which artifacts are resources because pull-node-version -> spec-node-version -> artifact_name-version
                        pass


        # Lineage-Edge Link Resources and Artifacts (FlorPlan Dag in Ground).

        # DEFER LINKING PULL TO pulled artifact



def commit(xp_state : State):
    assert False, "Deprecated Above Ground Commit: Do not call this method"
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
    assert False, "Deprecated above ground peek: Do not call this method"

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
    assert False, "Deprecated Above Groubnd Fork: Do not call this method"
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
    assert False, "Deprecated Above Ground Pull, do not call this method."
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