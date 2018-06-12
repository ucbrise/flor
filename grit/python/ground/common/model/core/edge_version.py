from ground.common.model.core.rich_version import RichVersion


class EdgeVersion(RichVersion):

    def __init__(self, json_payload):
        super().__init__(json_payload)

        self._edge_id = json_payload.get('edgeId')

        self._from_node_version_start_id = json_payload.get('fromNodeVersionStartId', 0)

        if json_payload.get('fromNodeVersionEndId', 0) <= 0:
            self._from_node_version_end_id = -1
        else:
            self._from_node_version_end_id = json_payload.get('fromNodeVersionEndId')

        self._to_node_version_start_id = json_payload.get('toNodeVersionStartId', 0)

        if json_payload.get('toNodeVersionEndId', 0) <= 0:
            self._to_node_version_end_id = -1
        else:
            self._to_node_version_end_id = json_payload.get('toNodeVersionEndId')

    @classmethod
    def from_edge_version(cls, _id, other_edge_version):
        return EdgeVersion.from_edge_version_and_rich_version(
            _id, other_edge_version, other_edge_version
        )

    @classmethod
    def from_edge_version_and_rich_version(cls, _id, other_rich_version, other_edge_version):
        return cls({
            'id': _id,
            'tags': other_rich_version.get_tags(),
            'structureVersionId': other_rich_version.get_structure_version_id(),
            'reference': other_rich_version.get_reference(),
            'referenceParameters': other_rich_version.get_parameters(),
            'edgeId': other_edge_version.get_edge_id(),
            'fromNodeVersionStartId': other_edge_version.get_from_node_version_start_id(),
            'fromNodeVersionEndId': other_edge_version.get_from_node_version_end_id(),
            'toNodeVersionStartId': other_edge_version.get_to_node_version_start_id(),
            'toNodeVersionEndId': other_edge_version.get_to_node_version_end_id(),
        })

    def get_edge_id(self):
        return self._edge_id

    def get_from_node_version_start_id(self):
        return self._from_node_version_start_id

    def get_from_node_version_end_id(self):
        return self._from_node_version_end_id

    def get_to_node_version_start_id(self):
        return self._to_node_version_start_id

    def get_to_node_version_end_id(self):
        return self._to_node_version_end_id

    def __eq__(self, other):
        if not isinstance(other, EdgeVersion):
            return False
        return (self._edge_id == other._edge_id
            and self._from_node_version_start_id == other._from_node_version_start_id
            and self._from_node_version_end_id == other._from_node_version_end_id
            and self._to_node_version_start_id == other._to_node_version_start_id
            and self._to_node_version_end_id == other._to_node_version_end_id
            and self.get_id() == other.get_id()
            and super().__eq__(other))
