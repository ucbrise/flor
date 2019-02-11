import ast, astor
import astunparse

import tempfile

import inspect
import os, sys

from uuid import uuid4
from typing import Callable, Any

from flor import util
from flor.controller.parser.visitor import Visitor
from flor.controller.parser.transformer import Transformer
from flor.controller.parser import injected
import flor.global_state as global_state
from flor.context.tree import Tree

from IPython.core.magic import register_cell_magic

import logging
import importlib.util
import shutil
from io import StringIO

logger = logging.getLogger(__name__)


class Mapper():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def map(self, path):
        assert len(path) >= len(self.src), "path: {}, self.src: {}".format(path, self.src)
        d = path[len(self.src) + 1 :]
        return os.path.join(self.dst, d)


if global_state.interactive:
    @register_cell_magic
    def florit(line, cell):
        """
        Cell magic

        :param line:
        :param cell:
        :return:
        """
        # TODO: Test on IPython (Tested on Jupyter)
        tree = ast.parse(cell)
        logger.debug(astor.dump_tree(tree))
        for idx, each in enumerate(tree.body):
            visitor = Visitor(False, os.path.abspath(util.get_notebook_name()))
            visitor.visit(each)
            visitor.consolidate_structs()

            transformer = Transformer(visitor.__structs__, [])
            transformer.visit(each)
            tree.body[idx] = each

        injected.log_record_buffer = []
        injected.log_record_flag = True

        shell = get_ipython().get_ipython()
        shell.run_cell(astunparse.unparse(tree))
        injected.file.close()
        injected.file = open(global_state.log_name, 'a')

        df = Tree(injected.log_record_buffer).get_df()

        injected.log_record_buffer = []
        injected.log_record_flag = False

        return df


def track(f: Callable[..., Any]):
    """
    Function decorator.

    Wrap a function that contains ``flor.log`` statements.
    Does program analysis to generate rich logs from log statements.

    Usage:

    .. code-block:: python

        @flor.track
        def foo():
            ...
            ... log.read('data.csv') ...
            ... log.param(10) ...
            ... log.metric(0.9) ...
            ... log.write('model.pkl') ...
            ...

    See: https://github.com/ucbrise/flor/blob/master/examples/logger/basic.py

    :param f: a function
    :return: a modified function that generates rich logs
    """
    # TODO: https://docs.python.org/3/library/code.html
    assert not global_state.interactive, "You're in an interactive environment, use %%florit instead"
    if global_state.ci_temporary_directory is None:
        global_state.ci_temporary_directory = tempfile.TemporaryDirectory()
        logger.debug("Temporary directory created at {}".format(global_state.ci_temporary_directory.name))

    filename = inspect.getsourcefile(f).split('/')[-1]

    func_name = f.__name__
    temp_dir = global_state.ci_temporary_directory.name

    secret_dir = os.path.join(temp_dir, '.secret')

    # First, copy the tree containing the executable file
    root_dir = os.path.dirname(inspect.getsourcefile(f))

    if not os.path.isdir(secret_dir):
        os.mkdir(secret_dir)

        mapper = Mapper(root_dir, temp_dir)
        global_state.mapper = mapper

        # TODO: Decide whether we want to follow links
        for path_name, sub_dirs, file_names in os.walk(root_dir):
            if '.git' in path_name:
                continue
            dest_path_name = mapper.map(path_name)
            for sub_dir in sub_dirs:
                if '.git' != sub_dir:
                    os.mkdir(os.path.join(dest_path_name, sub_dir))
            for f_name in file_names:
                if f_name[-len('.py'):] == '.py':
                    shutil.copy2(os.path.join(path_name, f_name), os.path.join(dest_path_name, f_name))

    tru_path = inspect.getsourcefile(f)
    dest_path = global_state.mapper.map(tru_path)

    if filename not in os.listdir(secret_dir):
        # Needs compilation
        with open(tru_path, 'r') as sourcefile:
            tree = ast.parse(sourcefile.read())
            # logger.debug(astor.dump_tree(tree))
        for idx, each in enumerate(tree.body):
            if (type(each) == ast.FunctionDef
                    and each.decorator_list
                    and astunparse.unparse(each.decorator_list[0]).strip() == 'flor.track'):
                logger.debug("Detected a Flor Track Execution decorator")
                visitor = Visitor(each.name, os.path.abspath(inspect.getsourcefile(f)))
                visitor.visit(each)
                visitor.consolidate_structs()

                args = [arg.arg for arg in each.args.args]

                transformer = Transformer(visitor.__structs__, args)
                transformer.visit(each)
                each.decorator_list = []
                tree.body[idx] = each
            if (type(each) == ast.With
                    and each.items
                    and astunparse.unparse(each.items[0].context_expr.func).strip() == 'flor.Context'):
                logger.debug("Detected a Flor Context")
                del tree.body[idx]
                
        def create_nested_dir(path):
            if not os.path.isdir(path):
                head, tail = os.path.split(path)
                os.mkdir(os.path.join(create_nested_dir(head), tail))
            return path
                
        try:
            with open(dest_path, 'w') as dest_f:
                dest_f.write(astunparse.unparse(tree))
                dest_f.write('\n')
        except FileNotFoundError:
            h, t = os.path.split(dest_path)
            create_nested_dir(h)
            with open(dest_path, 'w') as dest_f:
                dest_f.write(astunparse.unparse(tree))
                dest_f.write('\n')

        with open(os.path.join(secret_dir, filename), 'w') as dest_f:
            dest_f.write('\n')

    spec = importlib.util.spec_from_file_location("_{}".format(uuid4().hex), dest_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callable_function = getattr(module, func_name)
    del module

    return callable_function


