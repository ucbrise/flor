
class State:

    def __init__(self):
        self.depth_limit = None
        self.xp_name = None

def start():
    global shared_state
    shared_state = State()