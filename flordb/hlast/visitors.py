from _ast import AST, Constant
from typing import Any, Dict, Optional
import ast

from .. import utils


class WithExpVisitor(ast.NodeVisitor):
    """
    Signals to the replay planner that the script has a checkpointable scope.

    Pre-v4: the only signal was `with flor.checkpointing(...):`.
    v4: any `flor.loop` is also a checkpointable scope (piggy-backed via
    torch.save / torch.load hooks), so its presence is sufficient.
    """

    def __init__(self):
        super().__init__()
        self.found = False

    def visit_With(self, node: ast.With):
        pred = (
            isinstance(node.items[0].context_expr, ast.Call)
            and isinstance(node.items[0].context_expr.func, ast.Attribute)
            and isinstance(node.items[0].context_expr.func.value, ast.Name)
            and node.items[0].context_expr.func.value.id == "flor"
            and node.items[0].context_expr.func.attr == "checkpointing"
        )
        if pred:
            self.found = True
        else:
            self.generic_visit(node)

    def visit_For(self, node: ast.For):
        iter_s = ast.unparse(node.iter).strip()
        if iter_s.startswith("flor.loop"):
            self.found = True
        self.generic_visit(node)


class LoggedExpVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.names: Dict[str, int] = {}
        # Inverse of `names`. Not derivable from it: two flor.log calls may
        # share a name, and `names` keeps only the last lineno for those.
        # `--apply @LINENO` resolves through here.
        self.linenos: Dict[int, str] = {}

        self.line2level: Dict[int, int] = {}
        self.lvl = 0
        # loop_names[i] is the name of the flor.loop at nesting depth (i+1)
        # in the order they appear on the dominant nesting path. So for
        #   for epoch in flor.loop("epoch", ...):
        #       for step in flor.loop("step", ...):
        #           ...
        # we get loop_names == ["epoch", "step"].
        self.loop_names: list[str] = []

    def visit_For(self, node: ast.For):
        iter_s = ast.unparse(node.iter).strip()
        if iter_s.startswith("flor.loop"):
            start_lvl = self.lvl
            loop_name: Optional[str] = None
            try:
                call = node.iter
                if isinstance(call, ast.Call) and call.args:
                    first = call.args[0]
                    if isinstance(first, ast.Constant) and isinstance(
                        first.value, str
                    ):
                        loop_name = first.value
            except Exception:
                pass
            try:
                self.lvl += 1
                # Record the loop name at this depth on first encounter.
                if loop_name is not None and len(self.loop_names) < self.lvl:
                    self.loop_names.append(loop_name)
                self.generic_visit(node)
            finally:
                self.lvl = start_lvl
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        pred = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "flor"
            and node.func.attr == "log"
        )
        if not pred:
            return self.generic_visit(node)
        if len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
            self.names[str(node.args[0].value)] = node.lineno
            self.linenos[node.lineno] = str(node.args[0].value)
            self.line2level[node.lineno] = self.lvl
        else:
            raise IndexError("FLOR: Did you give flor.log a key? It takes 2 args.")

    def generic_visit(self, node: AST) -> Any:
        if hasattr(node, "lineno"):
            self.line2level[node.lineno] = self.lvl
        return super().generic_visit(node)


class ResumeBlockVisitor(ast.NodeVisitor):
    """
    Locate a PyTorch resume-from-checkpoint block at module scope.

    Two shapes. The keyed one, where a saved dict is unpacked by key:

        <lhs> = torch.load(<literal-path>)
        <target>.load_state_dict(<lhs>[<key>])
        ...

    and the flat one, where the file holds a single object's state outright:

        <target>.load_state_dict(torch.load(<literal-path>))

    The flat form pairs with `torch.save(model.state_dict(), path)` and is the
    more common of the two, so not matching it meant the most ordinary script
    in PyTorch was the one flor could say least about. It carries no key, which
    `applies` records as a key of None -- the whole file goes into that target.

    Captures path, lhs name (keyed form only), and the (target, key) pairs so
    flor.loop can auto-restore historical state on replay without
    `flor.checkpointing(...)` enrollment. Only module-scope matches are
    emitted; function/class-nested patterns are recorded in `unscoped_match` so
    the caller can warn.

    Inference stops at one checkpoint file. A block loading two different paths
    is recorded in `multi_path` and emits nothing: ResumeSpec addresses a single
    file, so picking one of them would silently restore half the state.
    """

    def __init__(self):
        super().__init__()
        self._depth = 0
        self.path: Optional[str] = None
        self.lhs_name: Optional[str] = None
        self.applies: list = []
        self.unscoped_match: bool = False
        self.multi_path: bool = False

    @property
    def found(self) -> bool:
        return (
            self.path is not None
            and bool(self.applies)
            and not self.multi_path
        )

    def _claim_path(self, path: str) -> bool:
        """Record the file this block resumes from; False if it's a second one."""
        if self.path is None:
            self.path = path
            return True
        if self.path != path:
            self.multi_path = True
            return False
        return True

    def visit_FunctionDef(self, node):
        self._depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self._depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._depth -= 1

    def visit_ClassDef(self, node):
        self._depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._depth -= 1

    def visit_Assign(self, node: ast.Assign):
        if self._is_torch_load_assign(node):
            if self._depth != 0:
                self.unscoped_match = True
            elif self.path is None:
                self._capture_load(node)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and self._is_load_state_dict_call(
            node.value
        ):
            if self._depth != 0:
                self.unscoped_match = True
            elif not self._capture_apply(node.value):
                # Not `<lhs>[<key>]`; try the flat `torch.load(...)` form.
                self._capture_flat_apply(node.value)
        self.generic_visit(node)

    @staticmethod
    def _is_torch_load_assign(node: ast.Assign) -> bool:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return False
        v = node.value
        return (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr == "load"
            and isinstance(v.func.value, ast.Name)
            and v.func.value.id == "torch"
        )

    @staticmethod
    def _is_load_state_dict_call(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load_state_dict"
            and isinstance(call.func.value, ast.Name)
        )

    @staticmethod
    def _literal_load_path(call: ast.Call) -> Optional[str]:
        """The literal path a `torch.load(...)` call reads, if it is literal.

        Dynamic paths fall back to `flor.restore(...)`: flor has to name the
        mirror file before the script runs, and only a literal lets it.
        """
        if not call.args:
            return None
        arg = call.args[0]
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, (str, bytes))):
            return None
        return str(arg.value) if isinstance(arg.value, str) else arg.value.decode()

    def _capture_load(self, node: ast.Assign):
        call = node.value
        if not isinstance(call, ast.Call):
            return
        path = self._literal_load_path(call)
        if path is None:
            return
        if not self._claim_path(path):
            return
        target0 = node.targets[0]
        assert isinstance(target0, ast.Name)
        self.lhs_name = target0.id

    def _capture_flat_apply(self, call: ast.Call) -> bool:
        """`<target>.load_state_dict(torch.load(<literal-path>))`.

        Recorded with a key of None: there is no dict to index into, the file
        *is* the target's state_dict.
        """
        if not call.args:
            return False
        arg = call.args[0]
        if not (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "load"
            and isinstance(arg.func.value, ast.Name)
            and arg.func.value.id == "torch"
        ):
            return False
        path = self._literal_load_path(arg)
        if path is None or not self._claim_path(path):
            return False
        assert isinstance(call.func, ast.Attribute) and isinstance(
            call.func.value, ast.Name
        )
        self.applies.append((call.func.value.id, None))
        return True

    def _capture_apply(self, call: ast.Call) -> bool:
        if self.lhs_name is None or not call.args:
            return False
        arg = call.args[0]
        if not isinstance(arg, ast.Subscript):
            return False
        if not (isinstance(arg.value, ast.Name) and arg.value.id == self.lhs_name):
            return False
        slice_node = arg.slice
        if isinstance(slice_node, ast.Constant):
            key = slice_node.value
        else:
            return False
        assert isinstance(call.func, ast.Attribute) and isinstance(
            call.func.value, ast.Name
        )
        self.applies.append((call.func.value.id, key))
        return True


class RestoreSignalVisitor(ast.NodeVisitor):
    """Whether the script says anything at all about how to restore state.

    ResumeBlockVisitor matches one narrow shape. A script that checkpoints with
    `torch.save` but declares its restore semantics some other way -- the flat
    `model.load_state_dict(torch.load(p))` one-liner, a computed path, a resume
    block inside a function -- leaves that visitor with nothing, and replay then
    quietly restores nothing at all. This collects the coarse signals needed to
    tell that silence apart from a script that simply has no checkpoints.
    """

    def __init__(self):
        super().__init__()
        self.torch_save = False
        self.enrolled = False  # `with flor.checkpointing(...)`
        self.declared = False  # `flor.restore(...)`

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "save" and isinstance(func.value, ast.Name):
                if func.value.id == "torch":
                    self.torch_save = True
            elif func.attr == "restore" and isinstance(func.value, ast.Name):
                self.declared = True
            elif func.attr == "checkpointing":
                # Matched as a call rather than off the `with` statement: the
                # context manager is the documented idiom, but entering it by
                # hand is still enrollment, and a false warning about a script
                # that *does* declare its restore semantics is worse than
                # missing an exotic one.
                self.enrolled = True
        self.generic_visit(node)


class NoGradVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.feeding = False
        self.names = {}
        self.tree = None

    def visit_Call(self, node: ast.Call):
        pred = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "flor"
            and node.func.attr == "log"
        )
        if not pred or not self.feeding:
            return self.generic_visit(node)
        if len(node.args) == 2 and isinstance(node.args[0], ast.Constant):
            self.names[str(node.args[0].value)] = node.lineno
        else:
            raise IndexError("FLOR: Did you give flor.log a key? It takes 2 args.")

    def visit_With(self, node: ast.With):
        if [True for each in node.items if "torch.no_grad" in ast.unparse(each)]:
            try:
                feeding = self.feeding
                self.feeding = True
                self.tree = node
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                self.feeding = feeding  # type: ignore


class NoGradTransformer(ast.NodeTransformer):
    def __init__(self, old_tree) -> None:
        super().__init__()
        self.their_tree = old_tree

    def visit_With(self, node: ast.With):
        if [True for each in node.items if "torch.no_grad" in ast.unparse(each)]:
            return self.generic_visit(self.their_tree)
        else:
            return self.generic_visit(node)
