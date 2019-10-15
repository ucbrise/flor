import pickle

class SerialWrapper():

    def __init__(self, x):
        self.data = x

    def get(self):
        return self.data

    def serialize(self):
        self.data = pickle.dumps(self.data)