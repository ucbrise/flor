import ast
import astor

import inspect
import os

from flor import util
from flor import global_state
from flor.controller.parser.visitor import Visitor

def track_execution(f):
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

    def callable_function(*args, **kwargs):

        function_parameters = inspect.signature(f).parameters

        for parameter_name in kwargs:
            assert parameter_name in function_parameters, \
                "Name {} is not a formal parameter of {}".format(parameter_name, func_name)

        assert len(args) <= f.__code__.co_argcount, \
            "Passed in more values than there are parameters in {}".format(func_name)

        for i, arg in enumerate(args):
            pos_var_name = f.__code__.co_varnames[0:f.__code__.co_argcount][i]
            assert pos_var_name not in kwargs, \
                "More than one value passed to formal parameter {} of {}".format(pos_var_name, func_name)
            kwargs[pos_var_name] = arg

        for parameter_name in function_parameters:
            if parameter_name not in kwargs:
                assert function_parameters[parameter_name].default is not inspect._empty, \
                    "Missing value for parameter {} in {}".format(parameter_name, func_name)
                kwargs[parameter_name] = function_parameters[parameter_name].default

        # kwargs ready

        tree = ast.parse(inspect.getsource(f))
        visitor = Visitor()
        visitor.visit(tree)
        structs = visitor.__structs__
        print('hi')



    return callable_function



