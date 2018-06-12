from ground.common.model.version.item import Item


class Edge(Item):

    def __init__(self, json_payload):
        super().__init__(json_payload)
        self._name = json_payload.get('name', '')
        self._from_node_id = str(json_payload.get('fromNodeId', 0))
        self._to_node_id = str(json_payload.get('toNodeId', 0))
        self._source_key = json_payload.get('sourceKey', '')

    @classmethod
    def from_edge(cls, _id, other_edge):
        return cls({
            'id': _id,
            'tags': other_edge.get_tags(),
            'fromNodeId': other_edge.get_from_node_id(),
            'toNodeId': other_edge.get_to_node_id(),
            'sourceKey': other_edge.get_source_key(),
        })

    def to_dict(self):
        d = {
            'id': self.get_id(),
            'name': self._name,
            'fromNodeId': self._from_node_id,
            'toNodeId': self._to_node_id,
            'sourceKey': self._source_key
        }
        if self.get_tags():
            d['tags'] = self.get_tags()

        return d

    def get_name(self):
        return self._name

    def get_from_node_id(self):
        return self._from_node_id

    def get_to_node_id(self):
        return self._to_node_id

    def get_source_key(self):
        return self._source_key

    def __eq__(self, other):
        return (
            isinstance(other, Edge)
            and self._name == other._name
            and self._source_key == other._source_key
            and self.get_id() == other.get_id()
            and self._from_node_id == other._from_node_id
            and self._to_node_id == other._to_node_id
            and self.get_tags() == other.get_tags()
        )
