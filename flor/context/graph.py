from typing import Union, Dict, Set
from flor.object_model.node import *

class Graph:

    def __init__(self):

        # Lineage Edges

        self.produces: Dict[Execution, Set[Union[Value, Reference]]] = {}
        self.is_produced_by: Dict[Union[Value, Reference], Set[Execution]] = {}

        self.consumes:  Dict[Execution, Set[Union[Value, Reference]]] = {}
        self.is_consumed_by: Dict[Union[Value, Reference], Set[Execution]] = {}

        self.invokes: Dict[Execution, Set[Execution]] = {}
        self.is_invoked_by: Dict[Execution, Set[Execution]] = {}

        self.contains: Dict[Execution, Set[Union[Value, Reference]]] = {}
        self.is_contained_in: Dict[Union[Value, Reference], Set[Execution]] = {}

        self.maps = {}
        self.is_mapped_from = {}

        self.bags = []

        # Instance variables

        self.starts: Set[Union[Value, Reference]] = set([])
        self.nodes: Set[Union[Execution, Reference, Value]] = set([])
        self.name_map : Dict[str, Set[Union[Execution, Reference, Value]]] = {}

    def node(self, v: Union[Value, Reference, Execution]):
        assert type(v) in {Value, Reference, Execution}, \
            "Invalid node type: '{}'. Expected Value, Reference, Execution".format(type(v))
        self.__put__(self.name_map, v.name, v)
        self.nodes |= {v,}
        if type(v) == Reference or type(v) == Value:
            self.starts |= {v, }

    def edge(self, u: Union[Value, Reference, Execution], v: Union[Value, Reference, Execution]):
        """
        Data-flow graph declaration interface
        If Execution E produces X
        and Execution E' consumes X
        then edge(E, X) and edge(X, E')
        :param u: source
        :param v: sink
        """
        assert u in self.nodes
        assert type(u) in {Value, Reference, Execution}, \
            "Invalid node type: '{}'. Expected Value, Reference, Execution".format(type(u))

        assert v in self.nodes
        assert type(v) in {Value, Reference, Execution}, \
            "Invalid node type: '{}'. Expected Value, Reference, Execution".format(type(v))

        assert not (type(u) in {Value, Reference} and type(v) in {Value, Reference}), \
            "Cannot draw an edge from {} to {}".format(type(u), type(v))

        self.starts -= {v, }

        if type(u) == Execution:
            if type(v) == Execution:
                self.__put__(self.invokes, u, v)
                self.__put__(self.is_invoked_by, v, u)
            elif type(v) == Reference or type(v) == Value:
                if v.stack_scoped:
                    self.__put__(self.contains, u, v)
                    self.__put__(self.is_contained_in, v, u)
                else:
                    self.__put__(self.produces, u, v)
                    self.__put__(self.is_produced_by, v, u)
        elif type(u) == Reference or type(u) == Value:
            assert type(v) == Execution
            self.__put__(self.is_consumed_by, u, v)
            self.__put__(self.consumes, v, u)

    def serialize(self):
        pass

    @staticmethod
    def deserialize():
        pass


    # Helpers

    @staticmethod
    def __put__(d, k, v):
        if k not in d:
            d[k] = {v, }
        else:
            d[k] |= {v, }
