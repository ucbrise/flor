from ground.common.model.core.rich_version import RichVersion


class NodeVersion(RichVersion):

    def __init__(self, json_payload):
        super().__init__(json_payload)

        self._node_id = str(json_payload.get('nodeId'))

    @classmethod
    def from_node_version(cls, _id, other_node_version):
        return NodeVersion.from_node_version_and_rich_version(
            _id, other_node_version, other_node_version
        )

    @classmethod
    def from_node_version_and_rich_version(cls, _id, other_rich_version, other_node_version):
        return cls({
            'id': _id,
            'tags': other_rich_version.get_tags(),
            'structureVersionId': other_rich_version.get_structure_version_id(),
            'reference': other_rich_version.get_reference(),
            'referenceParameters': other_rich_version.get_parameters(),
            'nodeId': other_node_version.get_node_id(),
        })

    def get_node_id(self):
        return self._node_id

    def __eq__(self, other):
        return (
            isinstance(other, NodeVersion)
            and self._node_id == other._node_id
            and super().__eq__(other)
        )
