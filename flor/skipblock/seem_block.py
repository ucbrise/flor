"""
Dummy SkipBlock Look Alike
For tracking loops (their static keys) that we refuse to transform
We use SeemBlocks for measuring loop runtime for all loops, helping us identify the outermost loop
"""

import time

class SeemBlock:

    def __init__(self, static_key, global_key=None):
        self.static_key = int(static_key)
        self.start_time = time.time()