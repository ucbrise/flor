from ground.common.model.version.item import Item


class Node(Item):

    def __init__(self, json_payload):
        super().__init__(json_payload)

        self._name = json_payload.get('name', '')
        self._source_key = json_payload.get('sourceKey', '')

    @classmethod
    def from_node(cls, node_id, other_node):
        return cls({
            'id': node_id,
            'tags': other_node.get_tags(),
            'name': other_node.get_name(),
            'sourceKey': other_node.get_source_key(),
        })

    def to_dict(self):
        d =  {
            'id': self.get_id(),
            'class': "Node",
            'name' : self._name,
            'sourceKey' : self._source_key
        }
        if self.get_tags():
            d['tags'] = self.get_tags()

        return d

    def get_item_id(self):
        return self.get_id()

    def get_name(self):
        return self._name

    def get_source_key(self):
        return self._source_key

    def __eq__(self, other):
        return (
            isinstance(other, Node)
            and self.get_name() == other.get_name()
            and self.get_source_key() == other.get_source_key()
            and self.get_item_id() == other._id
            and self.get_tags() == other._tags
        )
