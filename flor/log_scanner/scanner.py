import json
import pandas as pd

from .state_machines.actual_param import ActualParam
from .state_machines.root_expression import RootExpression
from .context import Ctx

class Scanner:

    def __init__(self, log_path):
        self.log_path = log_path
        self.state_machines = []

        self.trailing_ctx = None
        self.contexts = [] # Models current stack frame

        self.collected = []
        self.line_number = -1

    @staticmethod
    def is_subset(ours, theirs):
        """
        :param ours: A subset of self.contexts -- Most recent element at tail
        :param theirs: A subset of the Python stack found in log_record['catch_stack_frame'] -- Most recent element at head
        :return: Boolean
        """
        assert isinstance(ours, list)
        assert isinstance(theirs, list)

        if not ours:
            # An empty set is a subset of any set.
            return True
        if not theirs:
            # ours is not empty but theirs is
            return False

        try:
            func_name = ours[-1].func_ctx
            if func_name is not None:
                i = theirs.index(func_name)
                return Scanner.is_subset(ours[0:-1], theirs[i+1:])
            else:
                return Scanner.is_subset(ours[0:-1], theirs)
        except ValueError:
            # string is not in list
            return False

    def register_state_machine(self, fsm):
        self.state_machines.append(fsm)

    def register_ActualParam(self, *args, **kwargs):
        self.state_machines.append(ActualParam(*args, **kwargs))
        return self.state_machines[-1]

    def register_RootExpression(self, *args, **kwargs):
        self.state_machines.append(RootExpression(*args, **kwargs))
        return self.state_machines[-1]

    def scan(self, log_record):
        # log_record is a dictionary
        if 'session_start' in log_record or 'session_end' in log_record:
            return
        if 'file_path' in log_record:
            ctx = Ctx()
            ctx.parent_ctx = self.trailing_ctx
            self.trailing_ctx = ctx
            self.trailing_ctx.file_path = log_record['file_path']
            for fsm in self.state_machines:
                fsm.consume_lsn(log_record, self.trailing_ctx, self.contexts)
        elif 'class_name' in log_record:
            self.trailing_ctx.class_ctx = log_record['class_name']
            for fsm in self.state_machines:
                fsm.consume_lsn(log_record, self.trailing_ctx, self.contexts)
        elif 'start_function' in log_record:
            self.trailing_ctx.func_ctx = log_record['start_function']
            self.contexts.append(self.trailing_ctx)
            for fsm in self.state_machines:
                fsm.consume_lsn(log_record, self.trailing_ctx, self.contexts)
                if isinstance(fsm, ActualParam):
                    fsm.consume_func_name(log_record, self.trailing_ctx, self.contexts)
        elif 'end_function' in log_record:
            for fsm in self.state_machines:
                fsm.consume_lsn(log_record, self.trailing_ctx, self.contexts)
                if isinstance(fsm, ActualParam):
                    fsm.consume_func_name(log_record, self.trailing_ctx, self.contexts)
            ctx = self.contexts.pop()
            assert ctx.func_ctx == log_record['end_function'], \
                "For log record {} ... Expected: {}, Actual: {}".format(self.line_number,
                                                                        log_record['end_function'], ctx.func_ctx)
            if self.contexts:
                self.trailing_ctx = self.contexts[-1]
            else:
                while ctx.parent_ctx is not None:
                    ctx = ctx.parent_ctx
                self.trailing_ctx = ctx
        elif 'catch_stack_frame' in log_record:
            # Enfore that the Scanner Stack is a subset of the Python Stack
            # TODO: This approach will be an approximation unless we Flor-transform all of Python
            # Recursive solution to the problem.
            start_len = len(self.contexts)
            theirs = log_record['catch_stack_frame']
            while self.contexts and not self.is_subset(self.contexts, theirs):
                # Exception Raise/Catch are used to control flow. A raise can pop many items off the stack in one shot
                self.contexts.pop()
                if self.contexts:
                    self.trailing_ctx = self.contexts[-1]
                else:
                    self.trailing_ctx = None
            assert start_len == 0 or len(self.contexts) > 0, 'Could not align Scanner stack frame with Python stack frame'
        else:
            # Conditionally consume data log record
            for fsm in self.state_machines:
                out = fsm.consume_data(log_record, self.trailing_ctx, self.contexts)
                if out:
                    out = {fsm.name: list(out.values()).pop()}
                    self.collected.append({id(fsm): out})

    def scan_log(self):
        with open(self.log_path, 'r') as f:
            for idx, line in enumerate(f):
                self.line_number = idx + 1
                log_record = json.loads(line.strip())
                self.scan(log_record)

    def to_rows(self):
        #TODO: missing robust Dataflow analysis
        """
        x, y, z could be:
        *----*----*----*
        | x0 | .  | z0 |
        *----*----*----*
        | x0 | y0  | . |
        *----*----*----*

        or it could be:
        *----*----*----*
        | x0 | y0 | z0 |
        *----*----*----*

        ... Need to inspect loops and stack_frames to infer visibility and scopes over which the val is defined

        """
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
        # post-proc
        rows2 = []
        for row in rows:
            row2 = []
            for each in row:
                row2.append(list(each.values()).pop())
            rows2.append(row2)
        return rows2

    def to_df(self):
        rows = self.to_rows()
        d = {}
        for row in rows:
            for each in row:
                k = list(each.keys()).pop()
                if k not in d:
                    d[k] = [each[k],]
                else:
                    d[k].append(each[k])
        return pd.DataFrame(d)
