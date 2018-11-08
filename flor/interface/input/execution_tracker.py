import ast, astor
import astunparse

import tempfile

import inspect
import os, sys

from uuid import uuid4

from flor import util
from flor import global_state
from flor.controller.parser.visitor import Visitor
from flor.controller.parser.transformer import Transformer

import logging
import importlib.util
import shutil
logger = logging.getLogger(__name__)


class Mapper():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def map(self, path):
        assert len(path) >= len(self.src)
        d = path[len(self.src) + 1 :]
        return os.path.join(self.dst, d)


def track_execution(f):
    if global_state.ci_temporary_directory is None:
        global_state.ci_temporary_directory = tempfile.TemporaryDirectory()
        logger.debug("Temporary directory created at {}".format(global_state.ci_temporary_directory.name))

    if global_state.interactive:
        filename = inspect.getsourcefile(f).split('/')[-1]
        if '.py' not in filename[-3:]:
            if global_state.nb_name is None:
                filename =  os.path.basename(util.get_notebook_name())
            else:
                filename = global_state.nb_name
    else:
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
        for idx, each in enumerate(tree.body):
            if (type(each) == ast.FunctionDef
                    and each.decorator_list
                    and astunparse.unparse(each.decorator_list[0]).strip() == 'flor.track_execution'):
                logger.debug("Detected a Flor Track Executuon decorator")
                visitor = Visitor(each.name)
                visitor.visit(each)
                visitor.consolidate_structs()

                args = []
                for arg in each.args.args:
                    args.append(arg.arg)

                transformer = Transformer(visitor.__structs__, args)
                transformer.visit(each)
                each.decorator_list = []
                tree.body[idx] = each
            if (type(each) == ast.With
                    and each.items
                    and astunparse.unparse(each.items[0].context_expr.func).strip() == 'flor.Context'):
                logger.debug("Detected a Flor Context")
                del tree.body[idx]

        # print(astor.dump_tree(tree))
        # print(astunparse.unparse(tree))

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




# def depr_track_execution(f):
#     if global_state.interactive:
#         filename = inspect.getsourcefile(f).split('/')[-1]
#         if '.py' not in filename[-3:]:
#             if global_state.nb_name is None:
#                 filename =  os.path.basename(util.get_notebook_name())
#             else:
#                 filename = global_state.nb_name
#     else:
#         filename = inspect.getsourcefile(f).split('/')[-1]
#
#     func_name = f.__name__
#
#     def callable_function(*args, **kwargs):
#         #TODO: PARSE WHETHER OR NOT YOU EXEC
#         nonlocal func_name
#         function_parameters = inspect.signature(f).parameters
#
#         for parameter_name in kwargs:
#             assert parameter_name in function_parameters, \
#                 "Name {} is not a formal parameter of {}".format(parameter_name, func_name)
#
#         assert len(args) <= f.__code__.co_argcount, \
#             "Passed in more values than there are parameters in {}".format(func_name)
#
#         for i, arg in enumerate(args):
#             pos_var_name = f.__code__.co_varnames[0:f.__code__.co_argcount][i]
#             assert pos_var_name not in kwargs, \
#                 "More than one value passed to formal parameter {} of {}".format(pos_var_name, func_name)
#             kwargs[pos_var_name] = arg
#
#         for parameter_name in function_parameters:
#             if parameter_name not in kwargs:
#                 assert function_parameters[parameter_name].default is not inspect._empty, \
#                     "Missing value for parameter {} in {}".format(parameter_name, func_name)
#                 kwargs[parameter_name] = function_parameters[parameter_name].default
#
#         # kwargs ready
#
#         tree = ast.parse(inspect.getsource(f))
#         visitor = Visitor()
#         visitor.visit(tree)
#         visitor.consolidate_structs()
#
#         transformer = Transformer(visitor.__structs__, kwargs)
#         transformer.visit(tree)
#         tree.body[0].decorator_list = []
#
#         with tempfile.TemporaryDirectory() as d:
#             original_dir = os.getcwd()
#             os.chdir(d)
#             hex_name = "_{}".format(uuid4().hex)
#             unparsed = astunparse.unparse(tree)
#             global_state.tracked_executions[func_name] = unparsed
#             print(unparsed)
#             with open('{}.py'.format(hex_name), 'w') as g:
#                 g.write('import flor; log = flor.log\n')
#                 for func_name in global_state.tracked_executions:
#                     g.write(global_state.tracked_executions[func_name])
#                     g.write('\n')
#             sys.path.insert(0, d)
#             exec('import {}'.format(hex_name))
#         sys.path.pop(0)
#         os.chdir(original_dir)
#         eval('getattr({}, tree.body[0].name)(**kwargs)'.format(hex_name))
#         exec('del {}'.format(hex_name))
#
#     return callable_function


