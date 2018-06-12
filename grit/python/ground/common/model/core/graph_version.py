from ground.common.model.core.rich_version import RichVersion


class GraphVersion(RichVersion):

    def __init__(self, json_payload):
        super().__init__(json_payload)

        self._graph_id = json_payload.get('graphId')
        self._edge_version_ids = json_payload.get('edgeVersionIds') or []

    @classmethod
    def from_graph_version(cls, _id, other_graph_version):
        return GraphVersion.from_graph_version_and_rich_version(
            _id, other_graph_version, other_graph_version
        )

    @classmethod
    def from_graph_version_and_rich_version(cls, _id, other_rich_version, other_graph_version):
        return cls({
            'id': _id,
            'tags': other_rich_version.get_tags(),
            'structureVersionId': other_rich_version.get_structure_version_id(),
            'reference': other_rich_version.get_reference(),
            'referenceParameters': other_rich_version.get_parameters(),
            'graphId': other_graph_version.get_graph_id(),
            'edgeVersionIds': other_graph_version.get_edge_version_ids(),
        })

    def get_graph_id(self):
        return self._graph_id

    def get_edge_version_ids(self):
        return self._edge_version_ids

    def __eq__(self, other):
        return (
            isinstance(other, GraphVersion)
            and self._graph_id == other._graph_id
            and self.get_edge_version_ids() == other.get_edge_version_ids()
            and super().__eq__(other)
        )
