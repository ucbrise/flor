#!/usr/bin/env python3

from typing import List, Dict
from graphviz import Digraph

class MyDigraph:
    def __init__(self, name, node_attr=None, label=None):
        self.name = name
        self.node_attr = node_attr
        self.type = 'digraph'
        self.nodes = []
        self.edges = []
        self.subgraphs = []
        self.label = label

    def node(self, id, name, shape):
        self.nodes.append((id, name, shape))

    def edge(self, _from, _to):
        self.edges.append((_from, _to))

    def subgraph(self, g : 'MyDigraph'):
        g.type = 'subgraph'
        self.subgraphs.append(g)

    def write(self):
        with open('output.gv', 'w') as f:
            self.wrapped_write(f)

    def wrapped_write(self, f):
            f.write('{} {} {{\n'.format(self.type, self.name))
            if self.node_attr:
                f.write('node [shape=box]\n')
            if self.label:
                if 'x' in self.label:
                    f.write('label = "{}"'.format(self.label))
                else:
                    f.write('label = "{}x"'.format(self.label))
            for id, name, shape in self.nodes:
                f.write('\t {} [label="{}" shape={}]\n'.format(id, name, shape))
            for g, t in self.edges:
                f.write('\t{} -> {}\n'.format(g, t))
            for subgraph in self.subgraphs:
                subgraph.wrapped_write(f)
            f.write('}\n')




class VizNode:
    def __init__(self, id: str, name : str,
                 shape: str, next : List['VizNode'],
                 plate_label : str = None):
        self.id = id
        self.name = name
        self.shape = shape
        self.next = next
        self.plate_label = plate_label
        self.plate_label_source = None

    def get_children(self) -> List['VizNode']:
        return self.next

    def set_plate_label_source(self, name):
        self.plate_label_source = name

    def is_plate_collision(self, other: 'VizNode'):
        return (self.plate_label, self.plate_label_source) != (other.plate_label, other.plate_label_source)

    def print(self):
        out = "name: {}\n" \
              "shape: {}\n" \
              "next: {}\n" \
              "plate_label: {}\n" \
              "plate_label_source: {}\n".format(self.name, self.shape,
                                              [i.name for i in self.next], self.plate_label,
                                              self.plate_label_source)
        print(out)


class VizGraph:

    def __init__(self):
        # Start is a list of orphan Literal and Artifacts
        self.start : List[VizNode] = []
        self.dot = MyDigraph(name='source')
        self.counter = 0
        self.plate_heads = {}
        self.clusters = {
            (None, None) : self.dot
        }

    def bfs(self):
        explored = []
        queue = [i for i in self.start]
        visited = lambda : explored + queue

        while queue:
            node = queue.pop(0)
            # node.print()
            if (node.plate_label, node.plate_label_source) not in self.clusters:
                pl = node.plate_label
                pls = node.plate_label_source
                self.clusters[(pl, pls)] = MyDigraph(name='cluster' + pl+pls, node_attr={'shape': 'box'}, label=node.plate_label)

            if node.shape == 'underline':
                node.plate_label = None
                node.plate_label_source = None


            explored.append(node)

            for child in node.get_children():
                if child not in visited():
                    queue.append(child)

    def drawNodes(self):
        explored = []
        queue = [i for i in self.start]
        visited = lambda : explored + queue

        while queue:
            node = queue.pop(0)
            # node.print()

            dot = self.clusters[(node.plate_label, node.plate_label_source)]
            dot.node(node.id, node.name, shape=node.shape)

            explored.append(node)

            for child in node.get_children():
                if child not in visited():
                    queue.append(child)

    def drawEdges(self):
        explored = []
        queue = [i for i in self.start]
        visited = lambda: explored + queue

        subgraphed = []

        while queue:
            node = queue.pop(0)
            node.print()

            dot = self.clusters[(node.plate_label, node.plate_label_source)]

            explored.append(node)

            for child in node.get_children():
                subg = self.clusters[(child.plate_label, child.plate_label_source)]
                dot.edge(node.id, child.id)
                if dot != subg and dot.name + subg.name not in subgraphed:
                    dot.subgraph(subg)
                    subgraphed.append(dot.name + subg.name)

                if child not in visited():
                    queue.append(child)


    def to_graphViz(self):
        plate_heads = [i for i in self.plate_heads.keys()]
        for plate_head in plate_heads:
            self.derive_plate_membership(self.plate_heads[plate_head])

        self.bfs()
        self.drawNodes()
        self.drawEdges()
        self.dot.write()


    def rollback(self, v : VizNode):
        v = self.plate_heads[v.plate_label_source]
        explored = []
        stack = [v,]
        visited = lambda : explored + stack

        while stack:
            node = stack.pop()
            node.plate_label = None
            node.plate_label_source = None
            if node not in visited():
                explored.append(node)
                for child in node.get_children():
                    stack.append(child)

    def derive_plate_membership(self, v : VizNode):
        explored = []
        stack = [v,]
        visited = lambda : explored + stack

        while stack:
            node = stack.pop()
            if node not in visited():
                # node.print()
                explored.append(node)
                for child in node.get_children():
                    if not child.plate_label:
                        child.plate_label = node.plate_label
                        child.plate_label_source = node.plate_label_source
                    elif node.is_plate_collision(child):
                        a = child.plate_label
                        b = node.plate_label
                        c = a + 'x' + b
                        self.rollback(child)
                        self.rollback(node)
                        child.plate_label = c
                        child.plate_label_source = child.name
                        self.plate_heads[child.name] = child

                    stack.append(child)


    def newNode(self, name : str,
                 shape: str, next : List[VizNode],
                 plate_label : str = None) -> VizNode:
        id = self.counter
        self.counter += 1

        new_node = VizNode(str(id), name, shape, next, plate_label)

        if plate_label:
            assert name not in self.plate_heads
            self.plate_heads[name] = new_node
            new_node.set_plate_label_source(name)

        return new_node

    def regis_orphan(self, node: VizNode):
        self.start.append(node)


