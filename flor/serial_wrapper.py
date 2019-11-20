import cloudpickle
from torch import Tensor
import copy

class StateWrapper():
    #this only works for GPU state dictionaries because we don't have to worry about deep copies

    def __init__(self, x):
        self.data = x
        self.dict_check(self.data)

    def get(self):
        return self.data

    def dict_check(self, data):
        #if the item is a cpu tensor, we can't guarantee it won't be modified...
        #use get_device() to determine whether it is gpu or cpu
        for k,v in data.items():
            if isinstance(v, dict):
                self.dict_check(data[k])
            elif isinstance(v, list):
                self.list_check(data[k])
            elif isinstance(v, Tensor):
                data[k] = v.clone().cpu()

    def list_check(self, data):
        for x in range(len(data)):
            if isinstance(data[x], dict):
                self.dict_check(data[x])
            elif isinstance(data[x], list):
                self.list_check(data[x])
            elif isinstance(data[x], Tensor):
                data[x] = data[x].clone().cpu()

    def serialize(self):
        for k, v in self.data.items():
            self.data[k] = str(cloudpickle.dumps(v))
