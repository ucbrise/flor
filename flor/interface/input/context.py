class Context():
    def __init__(self, xp_name):
        self.xp_name = xp_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass