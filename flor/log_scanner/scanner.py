import json

from .scanners.actual_param import ActualParam
from .context import Ctx

# class Node:
#     def __init__(self, out, stack_frame):
#         self.out = out
#         self.children = []
#         self.parent = None
#         self.stack_frame = [i for i in stack_frame]

class Scanner:

    def __init__(self, log_path):
        self.log_path = log_path
        self.state_machines = []

        self.trailing_ctx = None
        self.contexts = [] # Models current stack frame

        self.collected = []
        self.nodes = []

        # # From old tree algorithm
        # self.marker = {}
        # self.stack_frames = []
        #
        # self.leaves = set([])
        #
        # ####
        # self.producers = {}

    # def _freeze_contexts(self, ctxs):
    #     return tuple([str(ctx) for ctx in ctxs])
    #
    # def _is_ancestor(self, stack_frame1, stack_frame2):
    #     if not stack_frame1 and not stack_frame2:
    #         return True
    #     top_stack_frame2 = stack_frame2[-1]
    #
    #     while top_stack_frame2 and top_stack_frame2 != stack_frame1[-1]:
    #         top_stack_frame2 = top_stack_frame2.parent_ctx
    #
    #     if not top_stack_frame2:
    #         return False
    #
    #     top_stack_frame1 = stack_frame1[-1]
    #
    #     while top_stack_frame2 and top_stack_frame1:
    #         if top_stack_frame1 != top_stack_frame2:
    #             return False
    #         top_stack_frame1 = top_stack_frame1.parent_ctx
    #         top_stack_frame2 = top_stack_frame2.parent_ctx
    #
    #     if top_stack_frame1 is None and top_stack_frame2 is None:
    #         return True
    #     return False

    def register_state_machine(self, fsm):
        self.state_machines.append(fsm)

    def scan(self, log_record):
        # log_record is a dictionary
        if 'session_start' in log_record or 'session_end' in log_record:
            return
        if 'file_path' in log_record:
            ctx = Ctx()
            ctx.parent_ctx = self.trailing_ctx
            self.trailing_ctx = ctx
            self.trailing_ctx.file_path = log_record['file_path']
        elif 'class_name' in log_record:
            self.trailing_ctx.class_ctx = log_record['class_name']
        elif 'start_function' in log_record:
            self.trailing_ctx.func_ctx = log_record['start_function']
            self.contexts.append(self.trailing_ctx)
            for fsm in self.state_machines:
                if isinstance(fsm, ActualParam):
                    fsm.consume_func_name(log_record, self.trailing_ctx, self.contexts)
        elif 'end_function' in log_record:
            for fsm in self.state_machines:
                if isinstance(fsm, ActualParam):
                    fsm.consume_func_name(log_record, self.trailing_ctx, self.contexts)
            ctx = self.contexts.pop()
            assert ctx.func_ctx == log_record['end_function']
            if self.contexts:
                self.trailing_ctx = self.contexts[-1]
            else:
                while ctx.parent_ctx is not None:
                    ctx = ctx.parent_ctx
                self.trailing_ctx = ctx
        else:
            # Conditionally consume data log record
            for fsm in self.state_machines:
                out = fsm.consume_data(log_record, self.trailing_ctx, self.contexts)
                if out:
                    self.collected.append(out)

    def scan_log(self):
        with open(self.log_path, 'r') as f:
            for line in f:
                log_record = json.loads(line.strip())
                self.scan(log_record)

    def to_rows(self):
        rows = []
        row = []
        for each in self.collected:
            k = list(each.keys()).pop()
            if k not in map(lambda x: list(x.keys()).pop(), row):
                row.append(each)
            else:
                rows.append(row)
                new_row = []
                for r in row:
                    if k not in r:
                        new_row.append(r)
                    else:
                        new_row.append(each)
                        break
                row = new_row
        rows.append(row)
        return rows

    # def proto_scan(self, log_record):
    #     if 'session_start' in log_record or 'session_end' in log_record:
    #         return
    #     if 'file_path' in log_record:
    #         ctx = Ctx()
    #         ctx.parent_ctx = self.trailing_ctx
    #         self.trailing_ctx = ctx
    #         self.trailing_ctx.file_path = log_record['file_path']
    #     elif 'class_name' in log_record:
    #         self.trailing_ctx.class_ctx = log_record['class_name']
    #     elif 'start_function' in log_record:
    #         self.trailing_ctx.func_ctx = log_record['start_function']
    #         self.contexts.append(self.trailing_ctx)
    #         for fsm in self.state_machines:
    #             if isinstance(fsm, ActualParam):
    #                 fsm.consume_func_name(log_record, self.trailing_ctx, self.contexts)
    #     elif 'end_function' in log_record:
    #         for fsm in self.state_machines:
    #             if isinstance(fsm, ActualParam):
    #                 fsm.consume_func_name(log_record, self.trailing_ctx, self.contexts)
    #         ctx = self.contexts.pop()
    #         assert ctx.func_ctx == log_record['end_function']
    #         if self.contexts:
    #             self.trailing_ctx = self.contexts[-1]
    #         else:
    #             while ctx.parent_ctx is not None:
    #                 ctx = ctx.parent_ctx
    #             self.trailing_ctx = ctx
    #     else:
    #         # Conditionally consume data log record
    #         for fsm in self.state_machines:
    #             out = fsm.consume_data(log_record, self.trailing_ctx, self.contexts)
    #             if out:
    #                 idx = len(self.nodes)
    #                 self.collected.append(out)
    #
    #                 node = Node(out, self.contexts)
    #                 self.nodes.append(node)
    #                 self.leaves |= {node,}
    #
    #                 if self._freeze_contexts(node.stack_frame) not in self.marker:
    #                     self.marker[self._freeze_contexts(node.stack_frame)] = node
    #                     if self.stack_frames:
    #                         for stack_frame in self.stack_frames:
    #                             if self._is_ancestor(stack_frame, node.stack_frame):
    #                                 parent = self.marker[self._freeze_contexts(stack_frame)]
    #                                 parent.children.append(node)
    #                                 node.parent = parent
    #                                 self.leaves -= {parent,}
    #                                 break
    #                     self.stack_frames.insert(0, node.stack_frame)
    #                 else:
    #                     if (node.stack_frame !=  self.nodes[-1].stack_frame and
    #                         self._is_ancestor(node.stack_frame, self.nodes[-1].stack_frame)):
    #                         self.marker = {}
    #                         self.stack_frames = [node.stack_frame,]
    #                     parent = self.nodes[idx - 1]
    #                     parent.children.append(node)
    #                     node.parent = parent
    #                     self.leaves -= {parent}
    #                     self.marker[self._freeze_contexts(node.stack_frame)] = node
