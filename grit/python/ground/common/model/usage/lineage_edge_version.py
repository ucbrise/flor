from ground.common.model.core.rich_version import RichVersion


class LineageEdgeVersion(RichVersion):

    def __init__(self, json_payload):
        super().__init__(json_payload)
        self._lineage_edge_id = json_payload.get('lineageEdgeId', 0)
        self._from_id = json_payload.get('fromId', 0)
        self._to_id = json_payload.get('toId', 0)

    @classmethod
    def from_lineage_version(cls, _id, other_lineage_version):
        return LineageEdgeVersion.from_lineage_version_and_rich_version(
            _id, other_lineage_version, other_lineage_version
        )

    @classmethod
    def from_lineage_version_and_rich_version(cls, _id, other_rich_version, other_lineage_version):
        return cls({
            'id': _id,
            'tags': other_rich_version.get_tags(),
            'structureVersionId': other_rich_version.get_structure_version_id(),
            'reference': other_rich_version.get_reference(),
            'referenceParameters': other_rich_version.get_parameters(),
            'lineageEdgeId': other_lineage_version.get_lineage_edge_id(),
            'fromId': other_lineage_version.get_from_id(),
            'toId': other_lineage_version.get_to_id(),
        })

    def get_lineage_edge_id(self):
        return self._lineage_edge_id

    def get_from_id(self):
        return self._from_id

    def get_to_id(self):
        return self._to_id

    def __eq__(self, other):
        return (
            isinstance(other, LineageEdgeVersion)
            and self._lineage_edge_id == other._lineage_edge_id
            and self._from_id == other._from_id
            and self._to_id == other._to_id
            and self.get_id() == other.get_id()
            and super().__eq__(other)
        )
