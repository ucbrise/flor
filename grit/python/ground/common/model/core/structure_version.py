from ground.common.model.version.version import Version


class StructureVersion(Version):

    def __init__(self, json_payload):
        super(StructureVersion, self).__init__(json_payload['id'])

        self._structure_id = json_payload.get('structureId')
        self._attributes = json_payload.get('attributes') or {}

    @classmethod
    def from_structure_version(cls, _id, other_structure_version):
        return StructureVersion.from_structure_version_and_rich_version(
            _id, other_structure_version, other_structure_version
        )

    @classmethod
    def from_structure_version_and_rich_version(cls, _id, other_rich_version, other_structure_version):
        return cls({
            'id': _id,
            'tags': other_rich_version.get_tags(),
            'structureVersionId': other_rich_version.get_structure_version_id(),
            'reference': other_rich_version.get_reference(),
            'referenceParameters': other_rich_version.get_parameters(),
            'structureId': other_structure_version.get_structure_id(),
        })

    def get_structure_id(self):
        return self._structure_id

    def get_attributes(self):
        return self._attributes

    def __eq__(self, other):
        return (
            isinstance(other, StructureVersion)
            and self.get_structure_id() == other.get_structure_id()
            and self.get_attributes() == other.get_attributes()
            and self.get_id() == other.get_id()
        )
