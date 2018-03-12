#!/usr/bin/env python3

from typing import List, Dict
from graphviz import Digraph
import numpy as np

"""
Should optimize with an all-pairs longest path algorithm.
Compute once, then query
"""


class MyDigraph:
    def __init__(self, name, node_attr=None, label=None):
        self.name = self.__format_name__(name)
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
        if g.name not in [i.name for i in self.subgraphs]:
            self.subgraphs.append(g)

    def __format_name__(self, name):
        if '.' in name:
            name = name.split('.')[0]
        return name

    def __clean_nesting__(self):
        these_subgraphs = [i.name for i in self.subgraphs]
        flattened_subgraphs = [g.__clean_nesting__() for g in self.subgraphs]
        if len(flattened_subgraphs) > 0:
            flattened_subgraphs = [item for sublist in flattened_subgraphs for item in sublist]
        self.subgraphs = [i for i in self.subgraphs if i.name not in flattened_subgraphs]
        return [i.name for i in self.subgraphs]


    def write(self):
        self.__clean_nesting__()
        with open('output.gv', 'w') as f:
            self.wrapped_write(f)

    def wrapped_write(self, f):
            f.write('{} {} {{\n'.format(self.type, self.name))
            if self.node_attr:
                f.write('node [shape=box]\n')
            if self.label:
                if 'x' in self.label:
                    f.write('label = "{}\n"'.format(self.label))
                else:
                    f.write('label = "{}x\n"'.format(self.label))
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

    def is_plate_collision(self, other: 'VizNode', graph: 'VizGraph'):
        if (self.plate_label, self.plate_label_source) != (other.plate_label, other.plate_label_source):
            our_path_rank = action_longest_path(graph, self, other)
            their_path_rank = action_longest_path(graph, other, other)
            if our_path_rank == their_path_rank:
                return 'COLLISION'
            elif our_path_rank < their_path_rank:
                return 'PARENT_WIN'
            else:
                return 'CHILD_WIN'
        return 'NO_COLLISION'

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
            # node.print()

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
        signature = (v.plate_label, v.plate_label_source)
        v = self.plate_heads[v.plate_label_source]
        explored = []
        stack = [v,]
        visited = lambda : explored + stack

        while stack:
            node = stack.pop()
            if (node.plate_label, node.plate_label_source) == signature:
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
                    else:
                        collision_type = node.is_plate_collision(child, self)
                        if collision_type == 'COLLISION':
                            a = child.plate_label
                            b = node.plate_label
                            c = a + 'x' + b
                            self.rollback(child)
                            self.rollback(node)
                            child.plate_label = c
                            child.plate_label_source = child.name
                            self.plate_heads[child.name] = child
                        elif collision_type == 'PARENT_WIN':
                            child.plate_label = node.plate_label
                            child.plate_label_source = node.plate_label_source

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


def action_longest_path(graph: VizGraph, s: VizNode, t: VizNode):
    s: VizNode = graph.plate_heads[s.plate_label_source]
    explored = []
    stack: List[VizNode] = [s, ]
    longest_paths = {}

    if s.shape == 'ellipse':
        longest_paths[s.name] = 1
    else:
        longest_paths[s.name] = 0

    while stack:
        node = stack.pop()
        explored.append(node)
        for child in node.get_children():

            if child.name not in longest_paths:
                if child.shape == 'ellipse':
                    longest_paths[child.name] = longest_paths[node.name] + 1
                else:
                    longest_paths[child.name] = longest_paths[node.name]
            else:
                if child.shape == 'ellipse':
                    longest_paths[child.name] = max(longest_paths[child.name], longest_paths[node.name] + 1)
                else:
                    longest_paths[child.name] = max(longest_paths[child.name], longest_paths[node.name])

            stack.append(child)
    return longest_paths[t.name]