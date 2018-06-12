import requests

import ground.common.model as model


class GroundClient:

    def __init__(self, hostname="localhost", port=9000):
        self.url = "http://" + hostname + ":" + str(port)

    '''
    HELPER METHODS
    '''

    def _make_get_request(self, endpoint, return_json=True):
        request = requests.get(self.url + endpoint)

        if return_json:
            try:
                if request.status_code >= 400:
                    return None
                return request.json()
            except ValueError:
                raise RuntimeError("Unexpected error: Could not decode JSON response from server. Response was " + str(request.status_code) + ".")
        else:
            pass

    def _make_post_request(self, endpoint, body, return_json=True):
        request = requests.post(self.url + endpoint, json=body)

        if return_json:
            try:
                if request.status_code >= 400:
                    return None
                return request.json()
            except ValueError:
                raise RuntimeError("Unexpected error: Could not decode JSON response from server. Response was " + str(request.status_code) + ".")
        else:
            pass

    def _get_rich_version_json(self, reference, reference_parameters, tags, structure_version_id, parent_ids):
        body = {}

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

    def _create_item(self, item_type, source_key, name, tags):
        endpoint = "/" + item_type
        body = {"sourceKey": source_key, "name": name}

        if tags:
            body["tags"] = tags

        return self._make_post_request(endpoint, body)

    def _get_item(self, item_type, source_key):
        return self._make_get_request("/" + item_type + "/" + source_key)

    def _get_item_latest_versions(self, item_type, source_key):
        return self._make_get_request("/" + item_type + "/" + source_key + "/latest")

    def _get_item_history(self, item_type, source_key):
        return self._make_get_request("/" + item_type + "/" + source_key + "/history")

    def _get_version(self, item_type, id):
        return self._make_get_request("/versions/" + item_type + "/" + str(id))

    '''
    EDGE METHODS
    '''

    def create_edge(self, source_key, name, from_node_id, to_node_id, tags=None):
        endpoint = "/edges"
        body = {"sourceKey": source_key, "name": name, "fromNodeId": from_node_id, "toNodeId": to_node_id}

        if tags:
            body["tags"] = tags

        response = self._make_post_request(endpoint, body)
        if response is None:
            return None
        else:
            return model.core.edge.Edge(response)

    def create_edge_version(self,
                            edge_id,
                            from_node_version_start_id,
                            to_node_version_start_id,
                            from_node_version_end_id=-1,
                            to_node_version_end_id=-1,
                            reference=None,
                            reference_parameters=None,
                            tags=None,
                            structure_version_id=-1,
                            parent_ids=None):

        endpoint = "/versions/edges"
        body = self._get_rich_version_json(reference, reference_parameters, tags, structure_version_id, parent_ids)

        body["edgeId"] = edge_id
        body["toNodeVersionStartId"] = to_node_version_start_id
        body["fromNodeVersionStartId"] = from_node_version_start_id

        if to_node_version_end_id > 0:
            body["toNodeVersionEndId"] = to_node_version_end_id

        if from_node_version_end_id > 0:
            body["fromNodeVersionEndId"] = from_node_version_end_id

        response = self._make_post_request(endpoint, body)
        if response is None:
            return None
        else:
            return model.core.edge_version.EdgeVersion(response)

    def get_edge(self, source_key):
        response = self._get_item("edges", source_key)
        if response is not None:
            return model.core.edge.Edge(response)

    def get_edge_latest_versions(self, source_key):
        return self._get_item_latest_versions("edges", source_key)

    def get_edge_history(self, source_key):
        return self._get_item_history("edges", source_key)

    def get_edge_version(self, id):
        response = self._get_version("edges", id)
        if response is not None:
            return model.core.edge_version.EdgeVersion(response)

    '''
    GRAPH METHODS
    '''

    def create_graph(self, source_key, name, tags=None):
        response = self._create_item("graphs", source_key, name, tags)
        if response is not None:
            return model.core.graph.Graph(response)

    def create_graph_version(self,
                             graph_id,
                             edge_version_ids,
                             reference=None,
                             reference_parameters=None,
                             tags=None,
                             structure_version_id=-1,
                             parent_ids=None):

        endpoint = "/versions/graphs"
        body = self._get_rich_version_json(reference, reference_parameters, tags, structure_version_id, parent_ids)

        body["graphId"] = graph_id
        body["edgeVersionIds"] = edge_version_ids

        response = self._make_post_request(endpoint, body)
        if response is not None:
            return model.core.graph_version.GraphVersion(response)

    def get_graph(self, source_key):
        response = self._get_item("graphs", source_key)
        if response is not None:
            return model.core.graph.Graph(response)

    def get_graph_latest_versions(self, source_key):
        return self._get_item_latest_versions("graphs", source_key)

    def get_graph_history(self, source_key):
        return self._get_item_history("graphs", source_key)

    def get_graph_version(self, id):
        response = self._get_version("graphs", id)
        if response is not None:
            return model.core.graph_version.GraphVersion(response)

    '''
    NODE METHODS
    '''

    def create_node(self, source_key, name, tags=None):
        response = self._create_item("nodes", source_key, name, tags)
        if response is not None:
            return model.core.node.Node(response)

    def create_node_version(self,
                            node_id,
                            reference=None,
                            reference_parameters=None,
                            tags=None,
                            structure_version_id=-1,
                            parent_ids=None):

        endpoint = "/versions/nodes"
        body = self._get_rich_version_json(reference, reference_parameters, tags, structure_version_id, parent_ids)

        body["nodeId"] = node_id

        response = self._make_post_request(endpoint, body)
        if response is not None:
            return model.core.node_version.NodeVersion(response)

    def get_node(self, source_key):
        return model.core.node.Node(self._get_item("nodes", source_key))

    def get_node_latest_versions(self, source_key):
        return self._get_item_latest_versions("nodes", source_key)

    def get_node_history(self, source_key):
        return self._get_item_history("nodes", source_key)

    def get_node_version(self, id):
        response = self._get_version("nodes", id)
        if response is not None:
            return model.core.node_version.NodeVersion(response)

    def get_node_version_adjacent_lineage(self, id):
        return self._make_get_request("/versions/nodes/adjacent/lineage/" + str(id))

    '''
    STRUCTURE METHODS
    '''

    def create_structure(self, source_key, name, tags=None):
        response = self._create_item("structures", source_key, name, tags)
        if response is not None:
            return model.core.structure.Structure(response)

    def create_structure_version(self,
                                 structure_id,
                                 attributes,
                                 parent_ids=None):

        if parent_ids is None:
            parent_ids = []

        endpoint = "/versions/structures"

        body = {
            "structureId": structure_id,
            "attributes": attributes,
            "parentIds": parent_ids
        }

        response = self._make_post_request(endpoint, body)
        if response is not None:
            return model.core.structure_version.StructureVersion(response)

    def get_structure(self, source_key):
        response = self._get_item("structures", source_key)
        if response is not None:
            return model.core.structure.Structure(response)

    def get_structure_latest_versions(self, source_key):
        return self._get_item_latest_versions("structures", source_key)

    def get_structure_history(self, source_key):
        return self._get_item_history("structures", source_key)

    def get_structure_version(self, id):
        response = self._get_version("structures", id)
        if response is not None:
            return model.core.structure_version.StructureVersion(response)

    '''
    LINEAGE EDGE METHODS
    '''

    def create_lineage_edge(self, source_key, name, tags=None):
        response = self._create_item("lineage_edges", source_key, name, tags)
        if response is not None:
            return model.usage.lineage_edge.LineageEdge(response)

    def create_lineage_edge_version(self,
                                    edge_id,
                                    to_rich_version_id,
                                    from_rich_version_id,
                                    reference=None,
                                    reference_parameters=None,
                                    tags=None,
                                    structure_version_id=-1,
                                    parent_ids=None):

        endpoint = "/versions/lineage_edges"
        body = self._get_rich_version_json(reference, reference_parameters, tags, structure_version_id, parent_ids)

        body["lineageEdgeId"] = edge_id
        body["toRichVersionId"] = to_rich_version_id
        body["fromRichVersionId"] = from_rich_version_id

        response = self._make_post_request(endpoint, body)
        if response is not None:
            return model.usage.lineage_edge_version.LineageEdgeVersion(response)

    def get_lineage_edge(self, source_key):
        response = self._get_item("lineage_edges", source_key)
        if response is not None:
            return model.usage.lineage_edge.LineageEdge(response)

    def get_lineage_edge_latest_versions(self, source_key):
        return self._get_item_latest_versions("lineage_edges", source_key)

    def get_lineage_edge_history(self, source_key):
        return self._get_item_history("lineage_edges", source_key)

    def get_lineage_edge_version(self, id):
        response = self._get_version("lineage_edges", id)
        if response is not None:
            return model.usage.lineage_edge_version.LineageEdgeVersion(response)

    '''
    LINEAGE GRAPH METHODS
    '''

    def create_lineage_graph(self, source_key, name, tags=None):
        response = self._create_item("lineage_graphs", source_key, name, tags)
        if response is not None:
            return model.usage.lineage_graph.LineageGraph(response)

    def create_lineage_graph_version(self,
                                     lineage_graph_id,
                                     lineage_edge_version_ids,
                                     reference=None,
                                     reference_parameters=None,
                                     tags=None,
                                     structure_version_id=-1,
                                     parent_ids=None):

        endpoint = "/versions/lineage_graphs"
        body = self._get_rich_version_json(reference, reference_parameters, tags, structure_version_id, parent_ids)

        body["lineageGraphId"] = lineage_graph_id
        body["lineageEdgeVersionIds"] = lineage_edge_version_ids

        response = self._make_post_request(endpoint, body)
        if response is not None:
            return model.usage.lineage_graph_version.LineageGraphVersion(response)

    def get_lineage_graph(self, source_key):
        response = self._get_item("lineage_graphs", source_key)
        if response is not None:
            return model.usage.lineage_graph.LineageGraph(response)

    def get_lineage_graph_latest_versions(self, source_key):
        return self._get_item_latest_versions("lineage_graphs", source_key)

    def get_lineage_graph_history(self, source_key):
        return self._get_item_history("lineage_graphs", source_key)

    def get_lineage_graph_version(self, id):
        response = self._get_version("lineage_graphs", id)
        if response is not None:
            return model.usage.lineage_graph_version.LineageGraphVersion(response)
