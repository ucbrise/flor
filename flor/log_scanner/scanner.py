import json

from .scanners.actual_param import ActualParam
from .scanners.root_expression import RootExpression
from .context import Ctx

class Scanner:

    def __init__(self, log_path):
        self.log_path = log_path
        self.state_machines = []

        self.trailing_ctx = None
        self.contexts = [] # Models current stack frame

        self.collected = []

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
                    self.collected.append({id(fsm): out})

    def scan_log(self):
        with open(self.log_path, 'r') as f:
            for line in f:
                log_record = json.loads(line.strip())
                self.scan(log_record)

    def to_rows(self):
        #TODO: missing robust Dataflow analysis
        """
        E
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
        return rows
