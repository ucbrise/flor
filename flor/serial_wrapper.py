import cloudpickle

class SerialWrapper():

    def __init__(self, x):
        self.data = x

    def get(self):
        return self.data

    def serialize(self):
        if isinstance(self.data, dict):
            for k, v in self.data.items():
                self.data[k] = str(cloudpickle.dumps(v))
        else:
            self.data = cloudpickle.dumps(self.data)