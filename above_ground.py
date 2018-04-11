#!/usr/bin/env python3

from typing import Set, Union
import hashlib
import json
import datetime

import jarvis.util as util
from jarvis.stateful import State
from jarvis.object_model import Artifact, Action, Literal


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

    # Begin

    sourcekeySpec = 'jarvis.' + xp_state.EXPERIMENT_NAME
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



def __tags_equal__(groundtag, mytag):
    groundtagprime = {}
    for kee in groundtag:
        groundtagprime[kee] = {}
        for kii in groundtag[kee]:
            if kii != 'id':
                groundtagprime[kee][kii] = groundtag[kee][kii]
    return groundtagprime == mytag

def newExperimentVersion(xp_state: State):
    # -- caution with fixed values like 'jarvisExperiment', allowing for early Ground Ref prototype

    # The name of this experiment is in a tag in the nodeVersion of 'jarvisExperiment'
    latest_experiment_node_versions = [x for x in xp_state.gc.getNodeLatestVersions('jarvisExperiment')
                                       if xp_state.gc.getNodeVersion(x).get_tags()['experimentName'][
                                           'value'] == xp_state.EXPERIMENT_NAME]

    # This experiment may have previous versions, then the most recents are the parents
    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisExperiment').get_id(),
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

    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisTrial').get_id(),
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

    candidate_nvs = [xp_state.gc.getNodeVersion(str(x)) for x in xp_state.gc.getNodeLatestVersions('jarvisLiteral')
                     if __tags_equal__(xp_state.gc.getNodeVersion(str(x)).get_tags(), my_tag)]
    assert len(candidate_nvs) <= 1

    if len(candidate_nvs) == 1:
        return candidate_nvs[0]
    else:
        return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisLiteral').get_id(),
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

    candidate_nvs = [xp_state.gc.getNodeVersion(str(x)) for x in xp_state.gc.getNodeLatestVersions('jarvisArtifact')
                     if __tags_equal__(xp_state.gc.getNodeVersion(str(x)).get_tags(), my_tag)]
    assert len(candidate_nvs) <= 1

    if len(candidate_nvs) == 1:
        return candidate_nvs[0]
    else:
        return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisArtifact').get_id(),
                                       tags=my_tag)

def newActionVersion(xp_state : State, actionName):
    my_tag = {     'actionName' : {
                         'key' : 'actionName',
                         'value' : actionName,
                         'type' : 'STRING'
                 }
    }
    return xp_state.gc.createNodeVersion(xp_state.gc.getNode('jarvisAction').get_id(),
                                         tags=my_tag)


def newExperimentTrialEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisExperimentjarvisTrial')

def newTrialLiteralEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisTrialjarvisLiteral')

def newTrialArtifactEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisTrialjarvisArtifact')

def newLiteralActionEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisLiteraljarvisAction')

def newArtifactActionEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisArtifactjarvisAction')

def newActionArtifactEdgeVersion(xp_state : State, fromNv, toNv):
    return __newEdgeVersion__(xp_state, fromNv, toNv, 'jarvisActionjarvisArtifact')

def __newEdgeVersion__(xp_state : State, fromNv, toNv, edgeKey):
    return xp_state.gc.createEdgeVersion(xp_state.gc.getEdge(edgeKey).get_id(),
                                         fromNv.get_id(),
                                         toNv.get_id())

