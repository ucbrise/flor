class VersionSuccessor:

    def __init__(self, id, from_id, to_id):
        self._id = id
        self._from_id = from_id
        self._to_id = to_id

    def get_id(self):
        return self._id

    def get_from_id(self):
        return self._from_id

    def get_to_id(self):
        return self._to_id
