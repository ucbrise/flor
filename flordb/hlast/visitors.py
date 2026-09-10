from _ast import AST, Constant
from typing import Any, Dict, Optional
import ast

from .. import utils


class WithExpVisitor(ast.NodeVisitor):
    """Whether the script has a flor.loop, the scope replay can narrow."""

    def __init__(self):
        super().__init__()
        self.found = False

    def visit_For(self, node: ast.For):
        if ast.unparse(node.iter).strip().startswith("flor.loop"):
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
    Locate a PyTorch resume-from-checkpoint block within one lexical scope.

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
    replay can turn the block into a no-op and recompute from the script's own
    initialization. Each function/class is scanned
    separately: a load in one scope must never match an apply in another.
    A nested match records its scope so replay can bind the live objects there,
    before that frame returns to a caller with different variable names.

    Inference stops at one checkpoint file. A block loading two different paths
    is recorded in `multi_path` and emits nothing: ResumeSpec addresses a single
    file, so neutralizing one of them would leave the other's load in place.
    """

    def __init__(self):
        super().__init__()
        self.scope_name: Optional[str] = None
        self.scope_lineno: Optional[int] = None
        self.lineno: Optional[int] = None
        self._nested = []
        self.ambiguous_scope = False
        self.path: Optional[str] = None
        self.lhs_name: Optional[str] = None
        self.applies: list = []
        self.multi_path: bool = False

    @property
    def found(self) -> bool:
        return (
            self.path is not None
            and bool(self.applies)
            and not self.multi_path
            and not self.ambiguous_scope
        )

    def visit_Module(self, node):
        self.generic_visit(node)
        candidates = ([self] if self.applies else []) + self._nested
        if any(c.multi_path for c in candidates):
            self.multi_path = True
        if len({c.path for c in candidates}) > 1:
            self.multi_path = True
        if len(candidates) > 1:
            self.ambiguous_scope = True
        elif candidates and candidates[0] is not self:
            match = candidates[0]
            for name in (
                "path", "lhs_name", "applies", "scope_name", "scope_lineno", "lineno"
            ):
                setattr(self, name, getattr(match, name))

    def _claim_path(self, path: str) -> bool:
        """Record the file this block resumes from; False if it's a second one."""
        if self.path is None:
            self.path = path
            return True
        if self.path != path:
            self.multi_path = True
            return False
        return True

    def _visit_scope(self, node):
        child = ResumeBlockVisitor()
        child.scope_name = node.name
        # Decorated functions' code objects start at the first decorator.
        child.scope_lineno = node.lineno
        if not isinstance(node, ast.ClassDef) and node.decorator_list:
            child.scope_lineno = node.decorator_list[0].lineno
        for stmt in node.body:
            child.visit(stmt)
        if child.applies or child.multi_path:
            self._nested.append(child)
        self._nested.extend(child._nested)

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_Assign(self, node: ast.Assign):
        if self._is_torch_load_assign(node):
            self._capture_load(node)
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and self._is_load_state_dict_call(
            node.value
        ):
            if not self._capture_apply(node.value):
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

        Replay recognizes the load it neutralizes by the file it names, and
        only a literal names one before the script runs.
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
        if self.lineno is None:
            self.lineno = node.lineno

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
        if self.lineno is None:
            self.lineno = call.lineno
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


class SetupLoadVisitor(ast.NodeVisitor):
    """Whether the script calls torch.load outside every flor.loop / iteration.

    That is where a resume block sits, and so where a load can put an earlier
    run's weights over the initialization replay recomputes from. A load inside
    a function counts too: the source doesn't say where the function is called.
    """

    def __init__(self):
        super().__init__()
        self.found = False
        self._depth = 0

    def _visit_body(self, node):
        self._depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._depth -= 1

    def visit_For(self, node: ast.For):
        if ast.unparse(node.iter).strip().startswith("flor.loop"):
            self._visit_body(node)
        else:
            self.generic_visit(node)

    def visit_With(self, node: ast.With):
        if any(
            ast.unparse(item.context_expr).strip().startswith("flor.iteration")
            for item in node.items
        ):
            self._visit_body(node)
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if (
            self._depth == 0
            and isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        ):
            self.found = True
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
