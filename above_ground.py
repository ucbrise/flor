#!/usr/bin/env python3

from typing import Set, Union
import hashlib
import json

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

    def stringify(v):
         # https://stackoverflow.com/a/22505259/9420936
        return hashlib.md5(json.dumps(str(v) , sort_keys=True).encode('utf-8')).hexdigest()

    sourcekeySpec = 'jarvis.' + xp_state.EXPERIMENT_NAME
    specnode = safeCreateGetNode(sourcekeySpec, "null")

    starts : Set[Union[Artifact, Literal]] = xp_state.eg.starts
    for node in starts:
        if type(node) == Literal:
            sourcekeyLit = sourcekeySpec + '.literal.' + node.name
            litnode = safeCreateGetNode(sourcekeyLit, "null")
            safeCreateGetEdge(sourcekeyLit, "null", specnode.get_id(), litnode.get_id())
            if node.__oneByOne__:
                for i, v in enumerate(node.v):
                    sourcekeyBind = sourcekeyLit + '.' + stringify(v)
                    bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                        'value':
                            {
                                'key': 'v',
                                'value': str(v),
                                'type' : 'STRING'
                            }})
                    safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
            else:
                sourcekeyBind = sourcekeyLit + '.' + stringify(node.v)
                bindnode = safeCreateGetNode(sourcekeyBind, "null", tags={
                    'value':
                        {
                            'key': 'v',
                            'value': str(node.v),
                            'type': 'STRING'
                        }})
                safeCreateGetEdge(sourcekeyBind, "null", litnode.get_id(), bindnode.get_id())
        elif type(node) == Artifact:
            sourcekeyArt = 'jarvis' + xp_state.EXPERIMENT_NAME + '.artifact.' + stringify(node.loc)
            artnode = safeCreateGetNode(sourcekeyArt, "null")
            safeCreateGetEdge(sourcekeyArt, "null", specnode.get_id(), artnode.get_id())
        else:
            raise TypeError(
                "Action cannot be in set: starts")




    # There is no self.xp_state



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

