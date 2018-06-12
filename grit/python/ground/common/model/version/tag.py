class Tag:

    def __init__(self, json_payload):
        self._id  = json_payload.get('id')
        self._key = json_payload.get('key')
        self._val = json_payload.get('value')

    def get_id(self):
        return self._id

    def get_key(self):
        return self._key

    def get_value(self):
        return self._val

    def __eq__(self, other):
        return (
            isinstance(other, Tag)
            and self._id == other._id
            and self._key == other._key
            and self._val == other._val
        )
