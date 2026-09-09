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
        self.loop_children = {}
        self._loops = []

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
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        iter_s = ast.unparse(node.iter).strip()
        if iter_s.startswith("flor.loop"):
            self.found = True
            call = node.iter
            name = None
            if call.args and isinstance(call.args[0], ast.Constant):
                if isinstance(call.args[0].value, str):
                    name = call.args[0].value
            if self._loops and self._loops[-1] is not None:
                self.loop_children.setdefault(self._loops[-1], []).append(name)
            self._loops.append(name)
            try:
                self.generic_visit(node)
            finally:
                self._loops.pop()
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node):
        outer = self._loops
        self._loops = []
        try:
            self.generic_visit(node)
        finally:
            self._loops = outer

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


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
    flor.loop can auto-restore historical state on replay without
    `flor.checkpointing(...)` enrollment. Each function/class is scanned
    separately: a load in one scope must never match an apply in another.
    A nested match records its scope so replay can bind the live objects there,
    before that frame returns to a caller with different variable names.

    Inference stops at one checkpoint file. A block loading two different paths
    is recorded in `multi_path` and emits nothing: ResumeSpec addresses a single
    file, so picking one of them would silently restore half the state.
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


class SaveShapeVisitor(ast.NodeVisitor):
    """Infer a checkpoint's layout from the `torch.save` call that writes it.

    ResumeBlockVisitor reads the *load* side, which is direct evidence: the
    script names the object each slice of the file goes into. A script that
    checkpoints but never resumes says nothing there -- and its save call
    carries the same information one step less directly:

        torch.save(net.state_dict(), "ckpt.pth")
            -> [("net", None)]
        torch.save({"model": net.state_dict(),
                    "optimizer": opt.state_dict()}, "ckpt.pth")
            -> [("net", "model"), ("opt", "optimizer")]

    One step less directly, because saving and restoring are not the same
    statement. `torch.save(best_model.state_dict(), p)` names the object the
    forward run saved, which need not be the object replay should load into,
    and nothing in the source separates the two: both are modules, both carry
    matching shapes, so a wrong guess here restores *successfully* into the
    wrong object instead of failing the way a bad target does. That is why this
    is the last inference tried and why the caller announces the mapping it
    deduced rather than applying it silently.

    Only saves sharing a scope with a `flor.loop` (or `flor.iteration`) are
    harvested: restore resolves these names against the frame the loop runs in,
    so a save inside a helper function names locals replay cannot reach. Such a
    save is recorded in `unscoped_match` so the caller can say so.

    Entries that aren't `<name>.state_dict()` are skipped rather than refused --
    the `"epoch"` and `"loss"` a checkpoint dict usually carries alongside are
    not restore targets. The rest of the shape rules follow ResumeBlockVisitor:
    one literal path, a second recorded in `multi_path` and emitting nothing.
    Flat and keyed saves of one path contradict each other, as do two flat saves
    of it, and land in `conflicting_shape`.
    """

    def __init__(self):
        super().__init__()
        # Innermost enclosing def/class per the walk; None is module scope.
        self._scopes: list = [None]
        self._loop_scopes: set = set()
        # (scope, path, applies, lineno) per matching torch.save.
        self._saves: list = []
        self.path: Optional[str] = None
        self.lineno: Optional[int] = None
        self.applies: list = []
        self.multi_path: bool = False
        self.conflicting_shape: bool = False
        self.unscoped_match: bool = False

    @property
    def found(self) -> bool:
        return self.path is not None and bool(self.applies)

    def visit_Module(self, node: ast.Module):
        # ast.NodeVisitor has no end-of-walk hook, and every result here is a
        # judgment over all the saves rather than any one of them.
        self.generic_visit(node)
        self._finalize()

    def _visit_scope(self, node):
        self._scopes.append(node)
        try:
            self.generic_visit(node)
        finally:
            self._scopes.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_For(self, node: ast.For):
        if ast.unparse(node.iter).strip().startswith("flor.loop"):
            self._loop_scopes.add(id(self._scopes[-1]))
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        for item in node.items:
            if ast.unparse(item.context_expr).strip().startswith("flor.iteration"):
                self._loop_scopes.add(id(self._scopes[-1]))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if self._is_torch_save(node) and node.args:
            path = self._literal_path(node)
            applies = self._applies_for(node.args[0])
            if path is not None and applies is not None:
                self._saves.append(
                    (id(self._scopes[-1]), path, applies, node.lineno)
                )
        self.generic_visit(node)

    @staticmethod
    def _is_torch_save(call: ast.Call) -> bool:
        return (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "save"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
        )

    @staticmethod
    def _literal_path(call: ast.Call) -> Optional[str]:
        """The literal path a `torch.save(...)` writes, if it is literal.

        Same requirement the load side has: flor names the mirror file before
        the script runs, and only a literal lets it.
        """
        if len(call.args) < 2:
            return None
        arg = call.args[1]
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, (str, bytes))):
            return None
        return str(arg.value) if isinstance(arg.value, str) else arg.value.decode()

    @staticmethod
    def _state_dict_owner(node) -> Optional[str]:
        """`net` for `net.state_dict()`, else None.

        The receiver has to be a plain name. `net.module.state_dict()` saves a
        DataParallel's inner module, and restoring it means unwrapping `net` the
        same way -- something the load side would have stated outright and this
        side can only assume.
        """
        if not isinstance(node, ast.Call) or node.args or node.keywords:
            return None
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "state_dict"
            and isinstance(func.value, ast.Name)
        ):
            return func.value.id
        return None

    @classmethod
    def _applies_for(cls, saved) -> Optional[list]:
        """The (target, key) pairs a saved object implies, or None if neither shape."""
        owner = cls._state_dict_owner(saved)
        if owner is not None:
            return [(owner, None)]
        if not isinstance(saved, ast.Dict):
            return None
        pairs = []
        for key, value in zip(saved.keys, saved.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            owner = cls._state_dict_owner(value)
            if owner is not None:
                pairs.append((owner, key.value))
        return pairs or None

    def _finalize(self):
        in_loop = [s for s in self._saves if s[0] in self._loop_scopes]
        if not in_loop:
            self.unscoped_match = bool(self._saves)
            return
        if len({path for _, path, _, _ in in_loop}) > 1:
            self.multi_path = True
            return
        self.path = in_loop[0][1]
        self.lineno = in_loop[0][3]
        applies: list = []
        for _, _, pairs, _ in in_loop:
            for pair in pairs:
                if pair not in applies:
                    applies.append(pair)
        # A flat save says the file *is* one object's state. Another save of the
        # same path -- keyed, or flat into a second object -- says it is
        # something else, and only the one that ran last is true. Two flat
        # *loads* of one file are consistent (ResumeBlockVisitor takes them);
        # two flat saves of it are not.
        if any(key is None for _, key in applies) and len(applies) > 1:
            self.conflicting_shape = True
            return
        self.applies = applies

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
