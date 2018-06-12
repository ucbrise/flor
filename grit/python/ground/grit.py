# /usr/bin/env python3
import requests
import json
import numpy as np
import os
import git
import subprocess
import time

from shutil import copyfile
# noinspection PyUnresolvedReferences
from ground.common.model.core.node import Node
# noinspection PyUnresolvedReferences
from ground.common.model.core.node_version import NodeVersion
# noinspection PyUnresolvedReferences
from ground.common.model.core.edge import Edge
# noinspection PyUnresolvedReferences
from ground.common.model.core.edge_version import EdgeVersion
# noinspection PyUnresolvedReferences
from ground.common.model.core.graph import Graph
# noinspection PyUnresolvedReferences
from ground.common.model.core.graph_version import GraphVersion
# noinspection PyUnresolvedReferences
from ground.common.model.core.structure import Structure
# noinspection PyUnresolvedReferences
from ground.common.model.core.structure_version import StructureVersion
# noinspection PyUnresolvedReferences
from ground.common.model.usage.lineage_edge import LineageEdge
# noinspection PyUnresolvedReferences
from ground.common.model.usage.lineage_edge_version import LineageEdgeVersion
# noinspection PyUnresolvedReferences
from ground.common.model.usage.lineage_graph import LineageGraph
# noinspection PyUnresolvedReferences
from ground.common.model.usage.lineage_graph_version import LineageGraphVersion
# noinspection PyUnresolvedReferences
from ground.common.model.version.tag import Tag

from . import globals
from . import gizzard

"""
Abstract class: do not instantiate
"""


class GroundAPI:
    headers = {"Content-type": "application/json"}

    ### EDGES ###
    def createEdge(self, sourceKey, fromNodeId, toNodeId, name="null", tags=None):
        d = {
            "sourceKey": sourceKey,
            "fromNodeId": fromNodeId,
            "toNodeId": toNodeId,
            "name": name
        }
        if tags is not None:
            d["tags"] = tags
        return d

    def createEdgeVersion(self, edgeId, toNodeVersionStartId, fromNodeVersionStartId, toNodeVersionEndId=None,
                          fromNodeVersionEndId=None, reference=None, referenceParameters=None, tags=None,
                          structureVersionId=None, parentIds=None):
        d = {
            "edgeId": edgeId,
            "fromNodeVersionStartId": fromNodeVersionStartId,
            "toNodeVersionStartId": toNodeVersionStartId
        }
        if toNodeVersionEndId is not None:
            d["toNodeVersionEndId"] = toNodeVersionEndId
        if fromNodeVersionEndId is not None:
            d["fromNodeVersionEndId"] = fromNodeVersionEndId
        if reference is not None:
            d["reference"] = reference
        if referenceParameters is not None:
            d["referenceParameters"] = referenceParameters
        if tags is not None:
            d["tags"] = tags
        if structureVersionId is not None:
            d["structureVersionId"] = structureVersionId
        if parentIds is not None:
            d["parentIds"] = parentIds
        return d

    def getEdge(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getEdge")

    def getEdgeLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeLatestVersions")

    def getEdgeHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeHistory")

    def getEdgeVersion(self, edgeId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getEdgeVersion")

    ### NODES ###
    def createNode(self, sourceKey, name="null", tags=None):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        if tags is not None:
            d["tags"] = tags
        return d

    def createNodeVersion(self, nodeId, reference=None, referenceParameters=None, tags=None,
                          structureVersionId=None, parentIds=None):
        d = {
            "nodeId": nodeId
        }
        if reference is not None:
            d["reference"] = reference
        if referenceParameters is not None:
            d["referenceParameters"] = referenceParameters
        if tags is not None:
            d["tags"] = tags
        if structureVersionId is not None:
            d["structureVersionId"] = structureVersionId
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

    def getNodeVersionAdjacentLineage(self, nodeId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getNodeVersionAdjacentLineage")

    ### GRAPHS ###
    def createGraph(self, sourceKey, name="null", tags=None):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        if tags is not None:
            d["tags"] = tags
        return d

    def createGraphVersion(self, graphId, edgeVersionIds, reference=None, referenceParameters=None,
                           tags=None, structureVersionId=None, parentIds=None):
        d = {
            "graphId": graphId,
            "edgeVersionIds": edgeVersionIds
        }
        if reference is not None:
            d["reference"] = reference
        if referenceParameters is not None:
            d["referenceParameters"] = referenceParameters
        if tags is not None:
            d["tags"] = tags
        if structureVersionId is not None:
            d["structureVersionId"] = structureVersionId
        if parentIds is not None:
            d["parentIds"] = parentIds
        return d

    def getGraph(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getGraph")

    def getGraphLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphLatestVersions")

    def getGraphHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphHistory")

    def getGraphVersion(self, graphId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getGraphVersion")

    ### STRUCTURES ###
    def createStructure(self, sourceKey, name="null", tags=None):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        if tags is not None:
            d["tags"] = tags
        return d

    def createStructureVersion(self, structureId, attributes, parentIds=None):
        d = {
            "structureId": structureId,
            "attributes": attributes
        }
        if parentIds is not None:
            d["parentIds"] = parentIds
        return d

    def getStructure(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getStructure")

    def getStructureLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getStructureLatestVersions")

    def getStructureHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getStructureHistory")

    def getStructureVersion(self, structureId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getStructureVersion")

    ### LINEAGE EDGES ###
    def createLineageEdge(self, sourceKey, name="null", tags=None):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        if tags is not None:
            d["tags"] = tags
        return d

    def createLineageEdgeVersion(self, lineageEdgeId, toRichVersionId, fromRichVersionId, reference=None,
                                 referenceParameters=None, tags=None, structureVersionId=None, parentIds=None):
        d = {
            "lineageEdgeId": lineageEdgeId,
            "toRichVersionId": toRichVersionId,
            "fromRichVersionId": fromRichVersionId
        }
        if reference is not None:
            d["reference"] = reference
        if referenceParameters is not None:
            d["referenceParameters"] = referenceParameters
        if tags is not None:
            d["tags"] = tags
        if structureVersionId is not None:
            d["structureVersionId"] = structureVersionId
        if parentIds is not None:
            d["parentIds"] = parentIds
        return d

    def getLineageEdge(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getLineageEdge")

    def getLineageEdgeLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getLineageEdgeLatestVersions")

    def getLineageEdgeHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getLineageEdgeHistory")

    def getLineageEdgeVersion(self, lineageEdgeId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getLineageEdgeVersion")

    ### LINEAGE GRAPHS ###
    def createLineageGraph(self, sourceKey, name="null", tags=None):
        d = {
            "sourceKey": sourceKey,
            "name": name
        }
        if tags is not None:
            d["tags"] = tags
        return d

    def createLineageGraphVersion(self, lineageGraphId, lineageEdgeVersionIds, reference=None,
                                  referenceParameters=None, tags=None, structureVersionId=None, parentIds=None):
        d = {
            "lineageGraphId": lineageGraphId,
            "lineageEdgeVersionIds": lineageEdgeVersionIds
        }
        if reference is not None:
            d["reference"] = reference
        if referenceParameters is not None:
            d["referenceParameters"] = referenceParameters
        if tags is not None:
            d["tags"] = tags
        if structureVersionId is not None:
            d["structureVersionId"] = structureVersionId
        if parentIds is not None:
            d["parentIds"] = parentIds
        return d

    def getLineageGraph(self, sourceKey):
        raise NotImplementedError("Invalid call to GroundClient.getLineageGraph")

    def getLineageGraphLatestVersions(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getLineageGraphLatestVersions")

    def getLineageGraphHistory(self, sourceKey):
        raise NotImplementedError(
            "Invalid call to GroundClient.getLineageGraphHistory")

    def getLineageGraphVersion(self, lineageGraphId):
        raise NotImplementedError(
            "Invalid call to GroundClient.getLineageGraphVersion")

class GitImplementation(GroundAPI):

    def __init__(self):

        self._items = ['edge', 'graph', 'node', 'structure', 'lineage_edge', 'lineage_graph', 'index']
        self.path = globals.GRIT_D

        self.cls2loc = {
            'Edge' : 'edge/',
            'edge': 'edge/',
            'EdgeVersion' : 'edge/',
            'edgeversion': 'edge/',
            'Graph' : 'graph/',
            'graph': 'graph/',
            'GraphVersion' : 'graph/',
            'graphversion': 'graph',
            'Node' : 'node/',
            'node': 'node/',
            'NodeVersion' : 'node/',
            'nodeversion': 'nodeversion/',
            'Structure' : 'structure/',
            'structure' : 'structure/',
            'StructureVersion' : 'structure/',
            'structureversion' : 'structure/',
            'LineageEdge' : 'lineage_edge/',
            'lineageedge' : 'lineage_edge/',
            'LineageEdgeVersion' : 'lineage_edge/',
            'lineageedgeversion' : 'lineage_edge/',
            'LineageGraph' : 'lineage_graph/',
            'lineagegraph' : 'lineage_graph/',
            'LineageGraphVersion' : 'lineage_graph/',
            'lineagegraphversion' : 'lineage_graph/'
        }

        if not os.path.isdir(self.path):
            os.mkdir(self.path)
            for item in self._items:
                os.mkdir(self.path + item)
        if not os.path.exists(self.path + 'next_id.txt'):
            with open(self.path + 'next_id.txt', 'w') as f:
                f.write('0')
        if not os.path.exists(self.path + 'index/' + 'index.json'):
            with open(self.path + 'index/' + 'index.json', 'w') as f:
                json.dump({}, f)
        if not os.path.exists(self.path + 'index/index_version.json'):
            with open(self.path + 'index/index_version.json', 'w') as f:
                json.dump({}, f)


    def _get_rich_version_json(self, item_type, reference, reference_parameters,
                               tags, structure_version_id, parent_ids):
        item_id = self._gen_id()
        body = {"id": item_id, "class": item_type}
        if reference:
            body["reference"] = reference

        if reference_parameters:
            body["referenceParameters"] = reference_parameters

        if tags:
            body["tags"] = tags

        if structure_version_id and int(structure_version_id) > 0:
            body["structureVersionId"] = structure_version_id

        if parent_ids:
            body["parentIds"] = parent_ids

        return body

    def _deconstruct_rich_version_json(self, body):
        # This method needs MORE TESTING
        bodyRet = dict(body)
        if "tags" in bodyRet and bodyRet["tags"]:
            bodyTags = {}
            for key, value in list((bodyRet["tags"]).items()):
                if isinstance(value, Tag):
                    bodyTags[key] = {'id': value.get_id(), 'key': value.get_key(), 'value': value.get_value()}
            bodyRet["tags"] = bodyTags

        return bodyRet

    def _create_item(self, item_type, source_key, name, tags):
        item_id = self._gen_id()
        body = {"sourceKey": source_key, "name": name, "class": item_type, "id": item_id}

        if tags:
            body["tags"] = tags

        return body

    def _deconstruct_item(self, item):
        body = {"id": item.get_id(), "class": type(item).__name__, "name": item.get_name(),
                "sourceKey": item.get_source_key()}

        if item.get_tags():
            bodyTags = {}
            for key, value in list((item.get_tags()).items()):
                if isinstance(value, Tag):
                    bodyTags[key] = {'id': value.get_id(), 'key': value.get_key(), 'value': value.get_value()}
            body["tags"] = bodyTags

        return body

    def _gen_id(self):
        with open(self.path + 'next_id.txt', 'r') as f:
            newid = int(f.read())
        nxtid = str(newid + 1)
        with open(self.path + 'next_id.txt', 'w') as f:
            f.write(nxtid)
        return str(newid)

    def _write_files(self, sourceKey, body, className):
        sourceKey = str(sourceKey)
        filename = sourceKey + '.json'
        route = os.path.join(self.path + self.cls2loc[className], sourceKey)

        if os.path.isdir(route):
            repo = git.Repo(route)
        else:
            repo = git.Repo.init(route)

        if 'Version' in className:
            vbody = body['ItemVersion']

            parents = []
            if 'parentIds' in vbody:
                parents = vbody['parentIds']

            if parents is None or len(parents) == 0:
                # Get root/first commit
                commit, id = gizzard.get_commits(sourceKey, self.cls2loc[className])[-1]
                detached_head = True

                for branch, commit_of_branch in gizzard.get_branch_commits(sourceKey, self.cls2loc[className]):
                    if commit_of_branch == commit:
                        gizzard.runThere(['git', 'checkout', branch], sourceKey, self.cls2loc[className])
                        detached_head = False
                        break

                if detached_head:
                    gizzard.runThere(['git', 'checkout', commit], sourceKey, self.cls2loc[className])
                    new_name = gizzard.new_branch_name(sourceKey, self.cls2loc[className])
                    gizzard.runThere(['git', 'checkout', '-b', new_name], sourceKey, self.cls2loc[className])

                # assert: Now at branch with head attached to first commit
            elif len(parents) == 1:
                commit = gizzard.id_to_commit(parents[0], sourceKey, self.cls2loc[className])
                detached_head = True

                for branch, commit_of_branch in gizzard.get_branch_commits(sourceKey, self.cls2loc[className]):
                    if commit_of_branch == commit:
                        gizzard.runThere(['git', 'checkout', branch], sourceKey, self.cls2loc[className])
                        detached_head = False
                        break

                if detached_head:
                    gizzard.runThere(['git', 'checkout', commit], sourceKey, self.cls2loc[className])
                    new_name = gizzard.new_branch_name(sourceKey, self.cls2loc[className])
                    gizzard.runThere(['git', 'checkout', '-b', new_name], sourceKey, self.cls2loc[className])

                # assert: Now at branch with head attached to some commit
            else:
                commits = [gizzard.id_to_commit(p, sourceKey, self.cls2loc[className]) for p in parents]
                branches = []

                for commit in commits:
                    detached_head = True

                    for branch, commit_of_branch in gizzard.get_branch_commits(sourceKey, self.cls2loc[className]):
                        if commit_of_branch == commit:
                            gizzard.runThere(['git', 'checkout', branch], sourceKey, self.cls2loc[className])
                            detached_head = False
                            branches.append(branch)
                            break

                    if detached_head:
                        gizzard.runThere(['git', 'checkout', commit], sourceKey, self.cls2loc[className])
                        new_name = gizzard.new_branch_name(sourceKey, self.cls2loc[className])
                        gizzard.runThere(['git', 'checkout', '-b', new_name], sourceKey, self.cls2loc[className])
                        branches.append(new_name)

                gizzard.runThere(['git', 'merge', '-s', 'ours', '-m', 'id: -1, class: Merge'] + branches[0:-1],
                                 sourceKey, self.cls2loc[className])
                gizzard.runThere(['git', 'branch', '-D'] + branches[0:-1], sourceKey, self.cls2loc[className])

        with open(os.path.join(route, filename), 'w') as f:
            json.dump(body, f)

        repo.index.add([os.path.join(route, filename)])

        if 'Version' in className:
            repo.index.commit("id: {}, class: {}".format(body["ItemVersion"]["id"], className))
        else:
            repo.index.commit("id: {}, class: {}".format(body["Item"]["id"], className))


    def _read_files(self, sourceKey, className, layer):
        route = os.path.join(self.path + self.cls2loc[className], sourceKey)
        with open(os.path.join(route, sourceKey + '.json'), 'r') as f:
            fileDict = json.load(f)
            fileDict = fileDict[layer]
            return fileDict


    def _read_version(self, id, className):
        files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        for file in files:
            filename = file.split('.')
            if (filename[-1] == 'json') and (filename[0] == str(id)):
                with open(self.path + file, 'r') as f:
                    fileDict = json.loads(f.read())
                    if (fileDict['class'] == className):
                        return fileDict

    def _read_all_version(self, sourceKey, className, baseClassName):
        baseId = (self._read_files(sourceKey, baseClassName))['id']
        baseIdName = baseClassName[:1].lower() + baseClassName[1:] + "Id"

        versions = {}
        files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        for file in files:
            filename = file.split('.')
            if (filename[-1] == 'json') and (filename[0] != 'ids'):
                with open(self.path + file, 'r') as f:
                    fileDict = json.loads(f.read())
                    if ((baseIdName in fileDict) and (fileDict[baseIdName] == baseId)
                        and (fileDict['class'] == className)):
                        versions[fileDict['id']] = fileDict
        return versions

    def _read_all_version_ever(self, className):
        versions = {}
        files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
        for file in files:
            filename = file.split('.')
            if (filename[-1] == 'json') and (filename[0] != 'ids'):
                with open(self.path + file, 'r') as f:
                    fileDict = json.loads(f.read())
                    if (fileDict['class'] == className):
                        versions[fileDict['id']] = fileDict
        return versions

    def _find_file(self, sourceKey, className):
        ruta = os.path.join(self.path + self.cls2loc[className], sourceKey)
        return os.path.isdir(ruta)

    def _map_index(self, id, sourceKey):
        # Sloppy index: must read full file to get mapping. Make clever
        # Also, assuming index can fit in memory
        id = str(id)
        ruta = self.path + 'index/'
        with open(ruta + 'index.json', 'r') as f:
            d = json.load(f)
        d[id] = sourceKey
        with open(ruta + 'index.json', 'w') as f:
            json.dump(d, f)

    def _map_version_index(self, id, sourceKey):
        id = str(id)
        ruta = self.path + 'index/'
        with open(ruta + 'index_version.json', 'r') as f:
            d = json.load(f)
        d[id] = sourceKey
        with open(ruta + 'index_version.json', 'w') as f:
            json.dump(d, f)


    def _read_map_index(self, id):
        # Sloppy index: must read full file to get mapping. Make clever
        # Also, assuming index can fit in memory
        id = str(id)
        ruta = self.path + 'index/'
        with open(ruta + 'index.json', 'r') as f:
            d = json.load(f)
        if id not in d:
            raise KeyError(
                "No such key in index: {}".format(id))
        return d[id]

    def _read_map_version_index(self, id):
        id = str(id)
        ruta = self.path + 'index/'
        with open(ruta + 'index_version.json', 'r') as f:
            d = json.load(f)
        if id not in d:
            raise KeyError(
                "No such key in index: {}".format(id))
        return d[id]

    def __run_proc__(self, bashCommand):
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        return str(output, 'UTF-8')

    def _commit(self, id, className, sourcekey):
        # Warning: Piggy-backing index file commit
        totFile = self.path + self.cls2loc[className] + sourcekey + '.json'
        self.repo.index.add([totFile, self.path + 'index/index.json', self.path + 'index/index_version.json'])
        self.repo.index.commit("id: " + str(id) + ", class: " + className)

    @staticmethod
    def _readable_to_conventional(filename):
        filename = filename.split('.')
        assert len(filename) == 3
        filename.reverse()
        filename = '.'.join(filename[1:])

        return filename

    @staticmethod
    def _conventional_to_readable(filename):
        filename = filename.split('.')
        assert len(filename) == 2
        filename.reverse()
        filename.append('txt')

        return '.'.join(filename)


    ### EDGES ###
    def createEdge(self, sourceKey, fromNodeId, toNodeId, name="null", tags=None):
        if not self._find_file(sourceKey, Edge.__name__):
            fromNodeId = str(fromNodeId)
            toNodeId = str(toNodeId)

            # Enforcing some integrity constraints
            self._read_map_index(fromNodeId)
            self._read_map_index(toNodeId)

            body = self._create_item(Edge.__name__, sourceKey, name, tags)
            body["fromNodeId"] = fromNodeId
            body["toNodeId"] = toNodeId
            edge = Edge(body)
            edgeId = str(edge.get_id())
            write = self._deconstruct_item(edge)
            write["fromNodeId"] = edge.get_from_node_id()
            write["toNodeId"] = edge.get_to_node_id()
            write = {"Item": write, "ItemVersion": {}}
            self._write_files(sourceKey, write, Edge.__name__)
            self._map_index(edgeId, sourceKey)
        else:
            raise FileExistsError(
                "Edge with source key '{}' already exists.".format(sourceKey))

        return edge

    def createEdgeVersion(self, edgeId, toNodeVersionStartId, fromNodeVersionStartId, toNodeVersionEndId=None,
                          fromNodeVersionEndId=None, reference=None, referenceParameters=None, tags=None,
                          structureVersionId=None, parentIds=None):

        # Missing integrity constraint checks:
        # Passed in node versions must be versions of nodes that are linked by an edge with edgeId == edgeId

        body = self._get_rich_version_json(EdgeVersion.__name__, reference, referenceParameters,
                                           tags, structureVersionId, parentIds)

        body["edgeId"] = str(edgeId)
        body["toNodeVersionStartId"] = str(toNodeVersionStartId)
        body["fromNodeVersionStartId"] = str(fromNodeVersionStartId)

        if toNodeVersionEndId and int(toNodeVersionEndId) > 0:
            body["toNodeVersionEndId"] = str(toNodeVersionEndId)

        if fromNodeVersionEndId and int(fromNodeVersionEndId) > 0:
            body["fromNodeVersionEndId"] = str(fromNodeVersionEndId)

        sourceKey = self._read_map_index(edgeId)
        edge = self.getEdge(sourceKey)

        edgeVersion = EdgeVersion(body)
        edgeVersionId = str(edgeVersion.get_id())

        write = self._deconstruct_rich_version_json(body)
        write = {"Item": edge.to_dict(), "ItemVersion": write}
        self._write_files(sourceKey, write, EdgeVersion.__name__)
        self._map_version_index(edgeVersionId, sourceKey)

        return edgeVersion

    def getEdge(self, sourceKey):
        if not self._find_file(sourceKey, Edge.__name__):
            raise FileNotFoundError(
                "Edge with source key '{}' does not exist.".format(sourceKey))
        return Edge(self._read_files(sourceKey, Edge.__name__, "Item"))


    def getEdgeLatestVersions(self, sourceKey):
        latest_versions = []
        for branch, commit in gizzard.get_branch_commits(sourceKey, 'edge'):
            gizzard.runThere(['git', 'checkout', branch], sourceKey, 'edge')
            readfiles = self._read_files(sourceKey, Edge.__name__, "ItemVersion")
            ev = EdgeVersion(readfiles)
            latest_versions.append(ev)

        return latest_versions

    def getEdgeHistory(self, sourceKey):
        return gizzard.gitdag(sourceKey, 'edge')

    def getEdgeVersion(self, edgeVersionId):
        sourceKey = self._read_map_version_index(edgeVersionId)
        for commit, id in gizzard.get_ver_commits(sourceKey, 'edge'):
            if id == int(edgeVersionId):
                with gizzard.chinto(os.path.join(globals.GRIT_D, 'edge', sourceKey)):
                    with gizzard.chkinto(commit):
                        readfiles = self._read_files(sourceKey, Edge.__name__, "ItemVersion")
                    ev = EdgeVersion(readfiles)
                return ev
        raise RuntimeError("Reached invalid line in getEdgeVersion")

    ### NODES ###
    def createNode(self, sourceKey, name="null", tags=None):
        if not self._find_file(sourceKey, Node.__name__):
            body = self._create_item(Node.__name__, sourceKey, name, tags)
            node = Node(body)
            nodeId = str(node.get_item_id())
            write = self._deconstruct_item(node)
            write = {"Item" : write, "ItemVersion": {}}
            self._write_files(sourceKey, write, Node.__name__)
            self._map_index(nodeId, sourceKey)
        else:
            raise FileExistsError(
                "Node with source key '{}' already exists.".format(sourceKey))

        return node

    def createNodeVersion(self, nodeId, reference=None, referenceParameters=None, tags=None,
                          structureVersionId=None, parentIds=None):
        body = self._get_rich_version_json(NodeVersion.__name__, reference, referenceParameters,
                                           tags, structureVersionId, parentIds)

        body["nodeId"] = str(nodeId)

        sourceKey = self._read_map_index(nodeId)
        node = self.getNode(sourceKey)

        nodeVersion = NodeVersion(body)
        nodeVersionId = str(nodeVersion.get_id())

        write = self._deconstruct_rich_version_json(body)
        write = {"Item": node.to_dict(), "ItemVersion": write}
        self._write_files(sourceKey, write, NodeVersion.__name__)
        self._map_version_index(nodeVersionId, sourceKey)
        return nodeVersion


    def getNode(self, sourceKey):
        if not self._find_file(sourceKey, Node.__name__):
            raise FileNotFoundError(
                "Node with source key '{}' does not exist.".format(sourceKey))
        return Node(self._read_files(sourceKey, Node.__name__, "Item"))

    def getNodeLatestVersions(self, sourceKey):
        latest_versions = []
        for branch, commit in gizzard.get_branch_commits(sourceKey, 'node'):
            gizzard.runThere(['git', 'checkout', branch], sourceKey, 'node')
            readfiles = self._read_files(sourceKey, Node.__name__, "ItemVersion")
            nv = NodeVersion(readfiles)
            latest_versions.append(nv)
        return latest_versions

    def getNodeHistory(self, sourceKey):
        return gizzard.gitdag(sourceKey, 'node')

    def getNodeVersion(self, nodeVersionId):
        sourceKey = self._read_map_version_index(nodeVersionId)
        for commit, id in gizzard.get_ver_commits(sourceKey, 'node'):
            if id == int(nodeVersionId):
                with gizzard.chinto(os.path.join(globals.GRIT_D, 'node', sourceKey)):
                    with gizzard.chkinto(commit):
                        readfiles = self._read_files(sourceKey, Node.__name__, "ItemVersion")
                    nv = NodeVersion(readfiles)
                return nv
        raise RuntimeError("Reached invalid line in getNodeVersion")

    def getNodeVersionAdjacentLineage(self, nodeVersionId):
        # All incoming and outgoing edges
        # Delaying implementation
        lineageEdgeVersionMap = self._read_all_version_ever(LineageEdgeVersion.__name__)
        lineageEdgeVersions = set(list(lineageEdgeVersionMap.keys()))
        adjacent = []
        for levId in lineageEdgeVersions:
            lev = lineageEdgeVersionMap[levId]
            if ((nodeVersionId == lev['toRichVersionId']) or (nodeVersionId == lev['fromRichVersionId'])):
                adjacent.append(lev)
        return adjacent


    ### GRAPHS ###
    def createGraph(self, sourceKey, name="null", tags=None):
        if not self._find_file(sourceKey, Graph.__name__, "Item"):
            body = self._create_item(Graph.__name__, sourceKey, name, tags)
            graph = Graph(body)
            graphId = graph.get_item_id()
            #self.graphs[sourceKey] = graph
            #self.graphs[graphId] = graph
            write = self._deconstruct_item(graph)
            self._write_files(graphId, write)
            self._commit(graphId, Graph.__name__)
        else:
            graph = self._read_files(sourceKey, Graph.__name__)
            graphId = graph['id']

        return graphId


    def createGraphVersion(self, graphId, edgeVersionIds, reference=None,
                           referenceParameters=None, tags=None, structureVersionId=None, parentIds=None):
        body = self._get_rich_version_json(GraphVersion.__name__, reference, referenceParameters,
                                           tags, structureVersionId, parentIds)

        body["graphId"] = graphId
        body["edgeVersionIds"] = edgeVersionIds

        graphVersion = GraphVersion(body)
        graphVersionId = graphVersion.get_id()

        #self.graphVersions[graphVersionId] = graphVersion

        write = self._deconstruct_rich_version_json(body)
        self._write_files(graphVersionId, write)
        self._commit(graphVersionId, GraphVersion.__name__)

        return graphVersionId

    def getGraph(self, sourceKey):
        return self._read_files(sourceKey, Graph.__name__)

    def getGraphLatestVersions(self, sourceKey):
        graphVersionMap = self._read_all_version(sourceKey, GraphVersion.__name__, Graph.__name__)
        graphVersions = set(list(graphVersionMap.keys()))
        is_parent = set([])
        for evId in graphVersions:
            ev = graphVersionMap[evId]
            if ('parentIds' in ev) and (ev['parentIds']):
                assert type(ev['parentIds']) == list
                for parentId in ev['parentIds']:
                    is_parent |= {parentId, }
        return [graphVersionMap[Id] for Id in list(graphVersions - is_parent)]

    def getGraphHistory(self, sourceKey):
        graphVersionMap = self._read_all_version(sourceKey, GraphVersion.__name__, Graph.__name__)
        graphVersions = set(list(graphVersionMap.keys()))
        parentChild = {}
        for evId in graphVersions:
            ev = graphVersionMap[evId]
            if ('parentIds' in ev) and (ev['parentIds']):
                assert type(ev['parentIds']) == list
                for parentId in ev['parentIds']:
                    if not parentChild:
                        graphId = ev['graphId']
                        parentChild[str(graphId)] = parentId
                    parentChild[str(parentId)] = ev['id']
        return parentChild

    def getGraphVersion(self, graphVersionId):
        return self._read_version(graphVersionId, GraphVersion.__name__)

    ### STRUCTURES ###
    def createStructure(self, sourceKey, name="null", tags=None):
        if not self._find_file(sourceKey, Structure.__name__):
            body = self._create_item(Structure.__name__, sourceKey, name, tags)
            structure = Structure(body)
            structureId = structure.get_item_id()
            #self.structures[sourceKey] = structure
            #self.structures[structureId] = structure
            write = self._deconstruct_item(structure)
            self._write_files(structureId, write)
            self._commit(structureId, Structure.__name__)
        else:
            structure = self._read_files(sourceKey, Structure.__name__)
            structureId = structure['id']

        return structureId


    def createStructureVersion(self, structureId, attributes, parentIds=None):
        body = {
            "id": self._gen_id(),
            "class":StructureVersion.__name__,
            "structureId": structureId,
            "attributes": attributes
        }

        if parentIds:
            body["parentIds"] = parentIds

        structureVersion = StructureVersion(body)
        structureVersionId = structureVersion.get_id()

        #self.structureVersions[structureVersionId] = structureVersion

        write = dict(body)
        self._write_files(structureVersionId, write)
        self._commit(structureVersionId, StructureVersion.__name__)

        return structureVersionId

    def getStructure(self, sourceKey):
        return self._read_files(sourceKey, Structure.__name__)

    def getStructureLatestVersions(self, sourceKey):
        structureVersionMap = self._read_all_version(sourceKey, StructureVersion.__name__, Structure.__name__)
        structureVersions = set(list(structureVersionMap.keys()))
        is_parent = set([])
        for evId in structureVersions:
            ev = structureVersionMap[evId]
            if ('parentIds' in ev) and (ev['parentIds']):
                assert type(ev['parentIds']) == list
                for parentId in ev['parentIds']:
                    is_parent |= {parentId, }
        return [structureVersionMap[Id] for Id in list(structureVersions - is_parent)]

    def getStructureHistory(self, sourceKey):
        structureVersionMap = self._read_all_version(sourceKey, StructureVersion.__name__, Structure.__name__)
        structureVersions = set(list(structureVersionMap.keys()))
        parentChild = {}
        for evId in structureVersions:
            ev = structureVersionMap[evId]
            if ('parentIds' in ev) and (ev['parentIds']):
                assert type(ev['parentIds']) == list
                for parentId in ev['parentIds']:
                    if not parentChild:
                        structureId = ev['structureId']
                        parentChild[str(structureId)] = parentId
                    parentChild[str(parentId)] = ev['id']
        return parentChild

    def getStructureVersion(self, structureVersionId):
        return self._read_version(structureVersionId, StructureVersion.__name__)


    ### LINEAGE EDGES ###
    def createLineageEdge(self, sourceKey, name="null", tags=None):
        if not self._find_file(sourceKey, LineageEdge.__name__):
            body = self._create_item(LineageEdge.__name__, sourceKey, name, tags)
            lineageEdge = LineageEdge(body)
            lineageEdgeId = str(lineageEdge.get_id())
            write = self._deconstruct_item(lineageEdge)
            write =  {"Item" : write, "ItemVersion": {}}
            self._write_files(sourceKey, write, LineageEdge.__name__)
            self._map_index(lineageEdgeId, sourceKey)
        else:
            raise FileExistsError(
                "Lineage Edge with source key '{}' already exists.".format(sourceKey))

        return lineageEdge


    def createLineageEdgeVersion(self, lineageEdgeId, toRichVersionId, fromRichVersionId, reference=None,
                                 referenceParameters=None, tags=None, structureVersionId=None, parentIds=None):
        body = self._get_rich_version_json(LineageEdgeVersion.__name__, reference, referenceParameters,
                                           tags, structureVersionId, parentIds)

        body["lineageEdgeId"] = lineageEdgeId
        body["toRichVersionId"] = toRichVersionId
        body["fromRichVersionId"] = fromRichVersionId

        sourceKey = self._read_map_index(lineageEdgeId)
        lineage_edge = self.getLineageEdge(sourceKey)

        lineageEdgeVersion = LineageEdgeVersion(body)
        lineageEdgeVersionId = str(lineageEdgeVersion.get_id())

        write = self._deconstruct_rich_version_json(body)
        write = {"Item": lineage_edge.to_dict(), "ItemVersion": write}
        self._write_files(sourceKey, write, LineageEdgeVersion.__name__)
        self._map_version_index(lineageEdgeVersionId, sourceKey)

        return lineageEdgeVersion

    def getLineageEdge(self, sourceKey):
        if not self._find_file(sourceKey, LineageEdge.__name__):
            raise FileNotFoundError(
                "Lineage Edge with source key '{}' does not exist".format(sourceKey))

        return LineageEdge(self._read_files(sourceKey, LineageEdge.__name__, "Item"))

    def getLineageEdgeLatestVersions(self, sourceKey):
        latest_versions = []
        for branch, commit in gizzard.get_branch_commits(sourceKey, 'lineage_edge'):
            gizzard.runThere(['git', 'checkout', branch], sourceKey, 'lineage_edge')
            readfiles = self._read_files(sourceKey, LineageEdge.__name__, "ItemVersion")
            lev = LineageEdgeVersion(readfiles)
            latest_versions.append(lev)
        return latest_versions

    def getLineageEdgeHistory(self, sourceKey):
        return gizzard.gitdag(sourceKey, 'lineage_edge')

    def getLineageEdgeVersion(self, lineageEdgeVersionId):
        sourceKey = self._read_map_version_index(lineageEdgeVersionId)
        for commit, id in gizzard.get_ver_commits(sourceKey, 'lineage_edge'):
            if id == int(lineageEdgeVersionId):
                with gizzard.chinto(os.path.join(globals.GRIT_D, 'lineage_edge', sourceKey)):
                    with gizzard.chkinto(commit):
                        readfiles = self._read_files(sourceKey, LineageEdge.__name__, "ItemVersion")
                    lev = LineageEdgeVersion(readfiles)
                return lev
        raise RuntimeError("Reached invalid line in getNodeVersion")

    ### LINEAGE GRAPHS ###
    def createLineageGraph(self, sourceKey, name="null", tags=None):
        if not self._find_file(sourceKey, LineageGraph.__name__):
            body = self._create_item(LineageGraph.__name__, sourceKey, name, tags)
            lineageGraph = LineageGraph(body)
            lineageGraphId = lineageGraph.get_id()
            #self.lineageGraphs[sourceKey] = lineageGraph
            #self.lineageGraphs[lineageGraphId] = lineageGraph
            write = self._deconstruct_item(lineageGraph)
            self._write_files(lineageGraphId, write)
            self._commit(lineageGraphId, LineageGraph.__name__)
        else:
            lineageGraph = self._read_files(sourceKey, LineageGraph.__name__)
            lineageGraphId = lineageGraph['id']

        return lineageGraphId


    def createLineageGraphVersion(self, lineageGraphId, lineageEdgeVersionIds, reference=None,
                                  referenceParameters=None, tags=None, structureVersionId=None, parentIds=None):
        body = self._get_rich_version_json(LineageGraphVersion.__name__, reference, referenceParameters,
                                           tags, structureVersionId, parentIds)

        body["lineageGraphId"] = lineageGraphId
        body["lineageEdgeVersionIds"] = lineageEdgeVersionIds

        lineageGraphVersion = LineageGraphVersion(body)
        lineageGraphVersionId = lineageGraphVersion.get_id()

        #self.lineageGraphVersions[lineageGraphVersionId] = lineageGraphVersion

        write = self._deconstruct_rich_version_json(body)
        self._write_files(lineageGraphVersionId, write)
        self._commit(lineageGraphVersionId, LineageGraphVersion.__name__)

        return lineageGraphVersionId

    def getLineageGraph(self, sourceKey):
        return self._read_files(sourceKey, LineageGraph.__name__)

    def getLineageGraphLatestVersions(self, sourceKey):
        lineageGraphVersionMap = self._read_all_version(sourceKey, LineageGraphVersion.__name__, LineageGraph.__name__)
        lineageGraphVersions = set(list(lineageGraphVersionMap.keys()))
        is_parent = set([])
        for evId in lineageGraphVersions:
            ev = lineageGraphVersionMap[evId]
            if ('parentIds' in ev) and (ev['parentIds']):
                assert type(ev['parentIds']) == list
                for parentId in ev['parentIds']:
                    is_parent |= {parentId, }
        return [lineageGraphVersionMap[Id] for Id in list(lineageGraphVersions - is_parent)]

    def getLineageGraphHistory(self, sourceKey):
        lineageGraphVersionMap = self._read_all_version(sourceKey, LineageGraphVersion.__name__, LineageGraph.__name__)
        lineageGraphVersions = set(list(lineageGraphVersionMap.keys()))
        parentChild = {}
        for evId in lineageGraphVersions:
            ev = lineageGraphVersionMap[evId]
            if ('parentIds' in ev) and (ev['parentIds']):
                assert type(ev['parentIds']) == list
                for parentId in ev['parentIds']:
                    if not parentChild:
                        lineageGraphId = ev['lineageGraphId']
                        parentChild[str(lineageGraphId)] = parentId
                    parentChild[str(parentId)] = ev['id']
        return parentChild

    def getLineageGraphVersion(self, lineageGraphVersionId):
        return self._read_version(lineageGraphVersionId, LineageGraphVersion.__name__)

    """
    def commit(self):
        stage = []
        for kee in self.graph.ids:
            if kee in self.graph.nodes:
                serial = self.graph.nodes[kee].to_json()
            elif kee in self.graph.nodeVersions:
                serial = self.graph.nodeVersions[kee].to_json()
            elif kee in self.graph.edges:
                serial = self.graph.edges[kee].to_json()
            elif kee in self.graph.edgeVersions:
                serial = self.graph.edgeVersions[kee].to_json()
            elif kee in self.graph.graphs:
                serial = self.graph.graphs[kee].to_json()
            elif kee in self.graph.graphVersions:
                serial = self.graph.graphVersions[kee].to_json()
            elif kee in self.graph.structures:
                serial = self.graph.structures[kee].to_json()
            elif kee in self.graph.structureVersions:
                serial = self.graph.structureVersions[kee].to_json()
            elif kee in self.graph.lineageEdges:
                serial = self.graph.lineageEdges[kee].to_json()
            elif kee in self.graph.lineageEdgeVersions:
                serial = self.graph.lineageEdgeVersions[kee].to_json()
            elif kee in self.graph.lineageGraphs:
                serial = self.graph.lineageGraphs[kee].to_json()
            else:
                serial = self.graph.lineageGraphVersions[kee].to_json()
            assert serial is not None
            with open(str(kee) + '.json', 'w') as f:
                f.write(serial)
            stage.append(str(kee) + '.json')
        repo = git.Repo.init(os.getcwd())
        repo.index.add(stage)
        repo.index.commit("ground commit")
        tree = repo.tree()
        with open('.jarvis', 'w') as f:
            for obj in tree:
                commithash = self.__run_proc__("git log " + obj.path).replace('\n', ' ').split()[1]
                if obj.path != '.jarvis':
                    f.write(obj.path + " " + commithash + "\n")
        repo.index.add(['.jarvis'])
        repo.index.commit('.jarvis commit')

    def load(self):
        if self.graph.ids:
            return
        os.chdir('../')

        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        listdir = [x for x in filter(is_number, os.listdir())]

        prevDir = str(len(listdir) - 1)
        os.chdir(prevDir)
        for _, _, filenames in os.walk('.'):
            for filename in filenames:
                filename = filename.split('.')
                if filename[-1] == 'json':
                    filename = '.'.join(filename)
                    with open(filename, 'r') as f:
                        self.to_class(json.loads(f.read()))
        os.chdir('../' + str(int(prevDir) + 1))
    """

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
