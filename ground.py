#/usr/bin/env python3
import requests
import json
import numpy as np
import os
import json
import git
import subprocess
from shutil import copyfile

class Node:

    def __init__(self, sourceKey=None, nodeId=None):
        self.sourceKey = sourceKey
        self.nodeId = nodeId
    
    def to_json(self):
        d = {
            'sourceKey': self.sourceKey,
            'nodeId': self.nodeId,
            'class': 'Node'
        }
        return json.dumps(d)

class NodeVersion:

    def __init__(self, node=Node()):
        self.sourceKey = node.sourceKey
        self.nodeId = node.nodeId
        self.tags = None
        self.parentIds = None
        self.nodeVersionId = None
    
    def to_json(self):
        d = {
            'sourceKey': self.sourceKey,
            'nodeId': self.nodeId,
            'tags': self.tags,
            'parentIds': self.parentIds,
            'nodeVersionId': self.nodeVersionId,
            'class': 'NodeVersion'
        }
        return json.dumps(d)


class Edge:

    def __init__(self, sourceKey=None, fromNodeId=None, toNodeId=None):
        self.sourceKey = sourceKey
        self.fromNodeId = fromNodeId
        self.toNodeId = toNodeId
        self.edgeId = None
    
    def to_json(self):
        d = {
            'sourceKey': self.sourceKey,
            'fromNodeId': self.fromNodeId,
            'toNodeId': self.toNodeId,
            'edgeId': self.edgeId,
            'class': 'Edge'
        }
        return json.dumps(d)

class EdgeVersion:

    def __init__(self, edge=Edge(), fromNodeVersionStartId=None, toNodeVersionStartId=None):
        self.sourceKey = edge.sourceKey
        self.fromNodeId = edge.fromNodeId
        self.toNodeId = edge.toNodeId
        self.edgeId = edge.edgeId
        self.fromNodeVersionStartId = fromNodeVersionStartId
        self.toNodeVersionStartId = toNodeVersionStartId
        self.edgeVersionId = None

    def to_json(self):
        d = {
            'sourceKey': self.sourceKey,
            'fromNodeId': self.fromNodeId,
            'toNodeId': self.toNodeId,
            'edgeId': self.edgeId,
            'fromNodeVersionStartId': self.fromNodeVersionStartId,
            'toNodeVersionStartId' : self.toNodeVersionStartId,
            'edgeVersionId': self.edgeVersionId,
            'class': 'EdgeVersion'
        }
        return json.dumps(d)

class Graph:

    def __init__(self):
        self.nodes = {}
        self.nodeVersions = {}
        self.edges = {}
        self.edgeVersions = {}
        self.ids = set([])

        self.__loclist__ = None
        self.__scriptNames__ = None

    def gen_id(self):
        newid = len(self.ids)
        self.ids |= {newid}
        return newid


"""
Abstract class: do not instantiate
"""
class GroundAPI:

    headers = {"Content-type": "application/json"}

    ### EDGES ###
    def createEdge(self, sourceKey, fromNodeId, toNodeId, name="null"):
        d = {
            "sourceKey": sourceKey,
            "fromNodeId": fromNodeId,
            "toNodeId": toNodeId,
            "name": name
        }
        return d

    def createEdgeVersion(self, edgeId, fromNodeVersionStartId, toNodeVersionStartId):
        d = {
            "edgeId": edgeId,
            "fromNodeVersionStartId": fromNodeVersionStartId,
            "toNodeVersionStartId": toNodeVersionStartId
        }
        return d

    def getEdge(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getEdge")

    def getEdgeLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeLatestVersions")

    def getEdgeHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeHistory")

    def getEdgeVersion(self, edgeVersionId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeVersion")

    ### NODES ###
    def createNode(self, sourceKey, name="null"):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        return d

    def createNodeVersion(self, nodeId, tags=None, parentIds=None):
        d = {
            "nodeId": nodeId
        }
        if tags is not None:
            d["tags"] = tags
        if parentIds is not None:
            d["parentIds"] = parentIds
        return d

    def getNode(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getNode")

    def getNodeLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeLatestVersions")

    def getNodeHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeHistory")

    def getNodeVersion(self, nodeId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeVersion")

    def getNodeVersionAdjacentLineage(self, nodeid):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeVersionAdjacentLineage")

    ### GRAPHS ###
    def createGraph(self, sourceKey, name="null"):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        return d

    def createGraphVersion(self, graphId, edgeVersionIds):
        d = {
            "graphId": graphId,
            "edgeVersionIds": edgeVersionIds
        }
        return d

    def getGraph(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getGraph")

    def getGraphLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphLatestVersions")

    def getGraphHitory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphHitory")

    def getGraphVersion(self, graphId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphVersionh")

    def commit(self, directory=None):
        return

    def load(self, directory):
        """
        This method is implemented by GitImplementation
        It is used to load the Ground Graph to memory (from filesystem)
        """
        return


class GitImplementation(GroundAPI):

    def __init__(self):
        self.graph = Graph()

    def __run_proc__(self, bashCommand):
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return str(output, 'UTF-8')

        ### EDGES ###
    def createEdge(self, sourceKey, fromNodeId, toNodeId, name="null"):
        edgeid = self.graph.gen_id()
        edge = Edge(sourceKey, fromNodeId, toNodeId)
        edge.edgeId = edgeid

        self.graph.edges[sourceKey] = edge
        self.graph.edges[edgeid] = edge

        return edgeid

    def createEdgeVersion(self, edgeId, fromNodeVersionStartId, toNodeVersionStartId):
        edge = self.graph.edges[edgeId]
        edgeVersion = EdgeVersion(edge, fromNodeVersionStartId, toNodeVersionStartId)

        edgeversionid = self.graph.gen_id()
        edgeVersion.edgeVersionId = edgeversionid

        if edgeVersion.sourceKey in self.graph.edgeVersions:
            self.graph.edgeVersions[edgeVersion.sourceKey].append(edgeVersion)
        else:
            self.graph.edgeVersions[edgeVersion.sourceKey] = [edgeVersion, ]
        self.graph.edgeVersions[edgeversionid] = edgeVersion
        return edgeversionid

    def getEdge(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getEdge")

    def getEdgeLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeLatestVersions")

    def getEdgeHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeHistory")

    def getEdgeVersion(self, edgeVersionId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeVersion")

    ### NODES ###
    def createNode(self, sourceKey, name="null"):
        if sourceKey not in self.graph.nodes:
            nodeid = self.graph.gen_id()
            node = Node(sourceKey, nodeid)


            self.graph.nodes[sourceKey] = node
            self.graph.nodes[nodeid] = node
        else:
            nodeid = self.graph.nodes[sourceKey].nodeId

        return nodeid

    def createNodeVersion(self, nodeId, tags=None, parentIds=None):
        node = self.graph.nodes[nodeId]
        nodeVersion = NodeVersion(node)
        if tags:
            nodeVersion.tags = tags
        if parentIds:
            nodeVersion.parentIds = parentIds
        else:
            # ALERT: THIS MAY NOT GENERALIZE TO K-LIFTING
            nlvs = self.getNodeLatestVersions(self.graph.nodes[nodeId].sourceKey)
            if nlvs:
                nodeVersion.parentIds = nlvs
            else:
                nodeVersion.parentIds = None

        nodeversionid = self.graph.gen_id()
        nodeVersion.nodeVersionId = nodeversionid

        if nodeVersion.sourceKey in self.graph.nodeVersions:
            self.graph.nodeVersions[nodeVersion.sourceKey].append(nodeVersion)
        else:
            self.graph.nodeVersions[nodeVersion.sourceKey] = [nodeVersion, ]
        self.graph.nodeVersions[nodeversionid] = nodeVersion
        return nodeversionid


    def getNode(self, sourceKey):
        return self.graph.nodes[sourceKey]

    def getNodeLatestVersions(self, sourceKey):
        assert sourceKey in self.graph.nodeVersions
        nodeVersions = set(self.graph.nodeVersions[sourceKey])
        is_parent = set([])
        for nv in nodeVersions:
            if nv.parentIds:
                assert type(nv.parentIds) == list
                for parentId in nv.parentIds:
                    is_parent |= {self.graph.nodeVersions[parentId],}
        return list(nodeVersions - is_parent)

    def getNodeHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeHistory")

    def getNodeVersion(self, nodeId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeVersion")

    def getNodeVersionAdjacentLineage(self, nodeid):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeVersionAdjacentLineage")

    ### GRAPHS ###
    def createGraph(self, sourceKey, name="null"):
        pass

    def createGraphVersion(self, graphId, edgeVersionIds):
        pass

    def getGraph(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getGraph")

    def getGraphLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphLatestVersions")

    def getGraphHitory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphHitory")

    def getGraphVersion(self, graphId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphVersionh")
    
    def commit(self, directory=None):
        if directory:
            os.makedirs(directory)
            for loc in self.graph.__loclist__:
                os.rename(loc, directory + '/' + loc)
            for script in self.graph.__scriptNames__:
                copyfile(script, directory + '/' + script)
            os.chdir(directory)
            stage = list(self.graph.__loclist__) + self.graph.__scriptNames__
            for kee in self.graph.ids:
                if kee in self.graph.nodes:
                    serial = self.graph.nodes[kee].to_json()
                elif kee in self.graph.nodeVersions:
                    serial = self.graph.nodeVersions[kee].to_json()
                elif kee in self.graph.edges:
                    serial = self.graph.edges[kee].to_json()
                else:
                    serial = self.graph.edgeVersions[kee].to_json()
                assert serial is not None
                with open(str(kee) + '.json', 'w') as f:
                    f.write(serial)
                stage.append(str(kee) + '.json')
            repo = git.Repo.init(os.getcwd())
            repo.index.add(stage)
            repo.index.commit("initial commit")
            tree = repo.tree()
            with open('.jarvis', 'w') as f:
                for obj in tree:
                    commithash = self.__run_proc__("git log " + obj.path).replace('\n', ' ').split()[1]
                    if obj.path != '.jarvis':
                        f.write(obj.path + " " + commithash + "\n")
            repo.index.add(['.jarvis'])
            repo.index.commit('.jarvis commit')
            os.chdir('../')
        else:
            pass

    def to_class(self, obj):
        if obj['class'] == 'Node':
            n = Node()
            n.sourceKey = obj['sourceKey']
            n.nodeId = obj['nodeId']

            self.graph.nodes[n.sourceKey] = n
            self.graph.nodes[n.nodeId] = n
            self.graph.ids |= {n.nodeId, }
        
        elif obj['class'] == 'NodeVersion':
            nv = NodeVersion()
            nv.sourceKey = obj['sourceKey']
            nv.nodeId = obj['nodeId']
            nv.tags = obj['tags']
            nv.parentIds = obj['parentIds']
            nv.nodeVersionId = obj['nodeVersionId']

            if nv.sourceKey in self.graph.nodeVersions:
                self.graph.nodeVersions[nv.sourceKey].append(nv)
            else:
                self.graph.nodeVersions[nv.sourceKey] = [nv, ]
            self.graph.nodeVersions[nv.nodeVersionId] = nv
            self.graph.ids |= {nv.nodeId, }
        
        elif obj['class'] == 'Edge':
            e = Edge()
            e.sourceKey = obj['sourceKey']
            e.fromNodeId = obj['fromNodeId']
            e.toNodeId =  obj['toNodeId']
            e.edgeId = obj['edgeId']





    def load(self, directory):
        if not os.path.exists(directory) or self.graph.ids:
            return
        os.chdir(directory)
        count = 0
        for _, _, filenames in os.walk('.'):
            for filename in filenames:
                filename = filename.split('.')
                if filename[-1] == 'json':
                    filename = '.'.join(filename)
                    with open(filename, 'r') as f:
                        self.to_class(json.loads(f.read()))
        
        os.chdir('../')

class GroundImplementation(GroundAPI):

    def __init__(self, host='localhost', port=9000):
        self.host = host
        self.port = str(port)
        self.url = "http://" + self.host + ":" + self.port

class GroundClient(GroundAPI):

    def __new__(*args, **kwargs):
        if args and args[1].strip().lower() == 'git':
            return GitImplementation(**kwargs)
        elif args and args[1].strip().lower() == 'ground':
            # EXAMPLE CALL: GroundClient('ground', host='localhost', port=9000)
            return GroundImplementation(**kwargs)
        else:
            raise ValueError(
                "Backend not supported. Please choose 'git' or 'ground'")
