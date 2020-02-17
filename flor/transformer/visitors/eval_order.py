import ast
import abc


class EvalOrderVisitor(ast.NodeVisitor, abc.ABC):

    @abc.abstractmethod
    def set_checkpoint(self):
        """
        Returns a copy of the  current state of the Node Visitor
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def branch_to(self, state):
        """
        counter-part to set_checkpoint
        Sets the Node Visitor state to the argument
        Returns the current state in case the caller wants to save it
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def merge_branches(self, state1, state2):
        raise NotImplementedError()

    @abc.abstractmethod
    def do_unimplemented_behavior(self):
        raise NotImplementedError()

    def visit_AnnAssign(self, node):
        self.do_unimplemented_behavior()

    def visit_Try(self, node):
        self.do_unimplemented_behavior()

    def visit_TryFinally(self, node):
        self.do_unimplemented_behavior()

    def visit_TryExcept(self, node):
        self.do_unimplemented_behavior()

    def visit_ExceptHandler(self, node):
        self.do_unimplemented_behavior()

    def visit_With(self, node):
        [self.visit(i) for i in node.items]
        [self.visit(s) for s in node.body]

    def visit_withitem(self, node):
        self.visit(node.context_expr)
        if node.optional_vars is not None:
            self.visit(node.optional_vars)

    def visit_Assign(self, node):
        self.visit(node.value)
        [self.visit(t) for t in node.targets]

    def visit_AugAssign(self, node):
        self.visit(node.value)
        self.visit(node.target)

    def visit_If(self, node):
        self.visit(node.test)
        state1 = self.set_checkpoint()
        [self.visit(s) for s in node.body]
        state2 = self.branch_to(state1)
        [self.visit(s) for s in node.orelse]
        self.merge_branches(state2, self.set_checkpoint())

    def visit_For(self, node):
        self.visit(node.iter)
        self.visit(node.target)
        [self.visit(s) for s in node.body]
        [self.visit(s) for s in node.orelse]

    def visit_While(self, node):
        self.visit(node.test)
        [self.visit(s) for s in node.body]
        [self.visit(s) for s in node.orelse]
