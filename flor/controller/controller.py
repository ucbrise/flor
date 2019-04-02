from flor.model import get, put
from flor.constants import *

class Controller:

    def __init__(self):
        self.depth_limit = get('depth_limit')
        if self.depth_limit is not Null:
            self.depth_limit -= 1
            put('depth_limit', self.depth_limit)

    def do(self, d):
        prev_depth_limit = self.depth_limit

        if 'end_function' in d:
            if self.depth_limit is not Null:
                self.depth_limit += 1
                put('depth_limit', self.depth_limit)

        if prev_depth_limit is not Null and prev_depth_limit < 0:
            return Exit
        else:
            return Continue

    def get_license_to_serialize(self):
        return self.depth_limit is Null or self.depth_limit >= 0