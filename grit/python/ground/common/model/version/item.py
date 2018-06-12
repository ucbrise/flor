from ground.common.model.version.tag import Tag

class Item:

    def __init__(self, json_payload):
        self._id   = str(json_payload.get('id')) or "0"
        self._tags = json_payload.get('tags') or {}

        for key, value in list(self._tags.items()):
            if not isinstance(value, Tag):
                self._tags[key] = Tag(value)

    def get_id(self):
        return str(self._id)

    def get_tags(self):
        return self._tags
