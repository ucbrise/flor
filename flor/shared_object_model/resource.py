from graphviz import Digraph, Source

from flor import util
from flor import viz
from flor.global_state import interactive

from flor.engine.executor import Executor
from flor.engine.expander import Expander
from flor.engine.consolidator import Consolidator
from flor.data_controller.organizer import Organizer
from flor.data_controller.versioner import Versioner
from flor.above_ground import PullTracker

from datetime import datetime

class Resource(object):

    def __init__(self, parent, xp_state):

        self.parent = parent

        if self.parent:
            self.parent.out_artifacts.append(self)

        self.xp_state = xp_state

    def getLocation(self):
        raise NotImplementedError("Abstract method Resource.getLocation must be overridden")

    def pull(self, manifest=None):
        raise NotImplementedError("Abstract method Resource.pull must be overridden")

    def peek(self, head=25, manifest=None, bindings=None, func = lambda x: x):
        raise NotImplementedError("Abstract method Resource.peek must be overridden")

    def __pull__(self, pulled_object, version=None):
        #Important: allows re-pulling without re-defining experiment
        self.xp_state.pre_pull = True

        assert Organizer.is_valid_version(pulled_object.xp_state, version), \
            "An experiment with the name '{}' exists".format(version)
        if version is None:
            self.write_version = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            version = str(version)
            self.write_version = version
        self.xp_state.pull_write_version = self.write_version
        experiment_graphs = Expander.expand(pulled_object.xp_state.eg, pulled_object)
        consolidated_graph = Consolidator.consolidate(experiment_graphs)
        Executor.execute(consolidated_graph)
        self.__plot__(pulled_object.name,
                      {'Art': 'box', 'Lit': 'underline'}[type(pulled_object).__name__[0:3]],
                      None, False)
        consolidated_graph.serialize()
        Versioner(self.write_version, consolidated_graph, pulled_object.xp_state).save_pull_event()
        # PullTracker(self.write_version, pulled_object.xp_state).pull(consolidated_graph)
        Organizer(self.write_version, consolidated_graph, pulled_object.xp_state).run()
        consolidated_graph.clean()
        self.xp_state.pre_pull = False



    def __plot__(self, nodename: str, shape: str, rankdir=None, view_now=True):
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
            if view_now:
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
            if not interactive:
                dot.render('output.gv', view=view_now)
            else:
                dot.render('output.gv', view=False)
                if view_now:
                    output_image = Source.from_file('output.gv')

        self.xp_state.eg.clean()
        return output_image