from typing import List, Dict, Any
import pandas as pd
import numpy as np

class Node:

    def __init__(self, d):
        self.assignee = d['assignee']
        self.caller = d['caller']
        self.from_arg = d['from_arg']
        self.in_execution = d['in_execution']
        self.in_file = d['in_file']
        self.instruction_no = d['instruction_no']
        self.keyword_name = d['keyword_name']
        self.pos = d['pos']
        self.runtime_value = d['runtime_value']
        self.typ = d['typ']
        self.value = d['value']
        self.stack_frame = d['__stack_frame__']

        self.children = []
        self.parent = None

class Tree:

    def __init__(self, log_records: List[Dict[str, Any]]):
        # set marker when I find child
        # clear marker when I find

        marker = {}
        stack_frames = []

        self.producers_table = {}
        self.leaves = set([])
        nodes : List[Node] = []

        for r in log_records:
            if 'from' in r and 'to' in r:
                if r['to'] not in self.producers_table:
                    self.producers_table[r['to']] = set(r['from'])
                else:
                    self.producers_table[r['to']] |= set(r['from'])
            else:
                nodes.append(Node(r))
                self.leaves |= {nodes[-1],}

        # print(self.producers_table)

        for idx, node in enumerate(nodes):
            # Attach this node to its parent, unless this node is root
            if node.stack_frame not in marker:
                # Update the marker and stack frames the first time we see a new stack_frame
                marker[node.stack_frame] = node
                if stack_frames:
                    for stack_frame in stack_frames:
                        if self.__is_ancestor__(stack_frame, node.stack_frame):
                            parent = marker[stack_frame]
                            parent.children.append(node)
                            node.parent = parent
                            self.leaves -= {parent,}
                            break
                stack_frames.insert(0, node.stack_frame)
            else:
                # Advance the marker
                if (node.stack_frame != nodes[-1].stack_frame and
                    self.__is_ancestor__(node.stack_frame, nodes[-1].stack_frame)):
                    marker = {}
                    stack_frames = [node.stack_frame]
                parent = nodes[idx - 1]
                parent.children.append(node)
                node.parent = parent
                self.leaves -= {parent,}
                marker[node.stack_frame] = node


    def get_df(self):
        # matrix[row_id][col_id] = v
        matrix = {}
        for col_id, leaf in enumerate(self.leaves):
            node = leaf
            while node:
                row_id = ' || '.join([str(node.assignee), str(node.keyword_name), str(node.value)])
                if row_id not in matrix:
                    matrix[row_id] = {}
                matrix[row_id][col_id] = node.runtime_value
                node = node.parent
        # len(d[row_id]) = len(self.leaves)
        d = {}
        for row_id in matrix:
            column = []
            for col_id in range(len(self.leaves)):
                if col_id not in matrix[row_id]:
                    column.append(np.nan)
                else:
                    column.append(matrix[row_id][col_id])
            d[row_id] = column

        return pd.DataFrame.from_dict(d, orient='index')

    def __is_ancestor__(self, stack_frame1, stack_frame2):
        if stack_frame1 == stack_frame2[0:len(stack_frame1)]:
            return True
        # stack_frame1 is not empty
        if stack_frame1[0:-1] == stack_frame2[0:-1]:
            # TODO: This is an overly-conservative lineage check, generalize
            n1, v1 = stack_frame1[-1]
            n2, v2 = stack_frame2[-1]
            if n1 == n2 and n1 == "function_body":
                return v1 in self.producers_table[v2]
        return False
