from graphviz import Digraph, Source

from flor import util
from flor import viz
from flor.global_state import interactive


class Resource(object):

    def __init__(self, parent, xp_state):

        self.parent = parent

        if self.parent:
            self.parent.out_artifacts.append(self)

        self.xp_state = xp_state

    def getLocation(self):
        raise NotImplementedError("Abstract method Resource.getLocation must be overridden")

    def pull(self, manifest=None):
        pass

    def peek(self, head=25, manifest=None, bindings=None, func = lambda x: x):
        pass

    def __plot__(self, nodename: str, shape: str, rankdir=None):
        """
        Acceptable to leave here
        :param rankdir: The visual direction of the DAG
        """
        # Prep globals, passed through arguments

        self.xp_state.eg.serialize()

        self.xp_state.nodes = {}
        self.xp_state.edges = []

        dot = Digraph()
        output_image = None
        # diagram = {"dot": dot, "counter": 0, "sha": {}}

        if not util.isOrphan(self):
            # self.parent.__plotWalk__(diagram)
            vg = viz.VizGraph()
            self.parent.__plotWalk__(vg)
            # vg.bft()
            vg.to_graphViz()
            if not interactive:
                Source.from_file('output.gv').view()
            else:
                output_image = Source.from_file('output.gv')
        else:
            node_diagram_id = '0'

            dot.node(node_diagram_id, nodename, shape=shape)
            self.xp_state.nodes[nodename] = node_diagram_id
            dot.format = 'png'
            if rankdir == 'LR':
                dot.attr(rankdir='LR')
            dot.render('driver.gv', view=True)

        self.xp_state.eg.clean()
        return output_image