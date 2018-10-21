#!/usr/bin/env python3

import inspect

import ast
import os

from flor import util
from flor import global_state

from flor.interface.input.experiment import Experiment


def func(foo):
    """
    Function decorator
    :param foo: Function
    """
    if global_state.interactive:
        filename = inspect.getsourcefile(foo).split('/')[-1]
        if '.py' not in filename[-3:]:
            if global_state.nb_name is None:
                filename =  os.path.basename(util.get_notebook_name())
            else:
                filename = global_state.nb_name
    else:
        filename = inspect.getsourcefile(foo).split('/')[-1]

    return filename, foo.__name__, foo

def track_action(xp_name):
    """
    Auto-generates flor Graph
    Single Action. Many literals in, Many literals out
    Artifacts not supported
    :param xp_name:
    :return:
    """

    def function_processor(f):
        tru_f = func(f)

        def callable_function(*pos, **kws):

            # INPUTS

            pos_var_names = [i for i in f.__code__.co_varnames if i != 'kwargs']
            for kee in kws:
                assert kee in f.__code__.co_varnames
            d = {}
            d.update(kws)
            for i, positional_arg in enumerate(pos):
                kee = pos_var_names[i]
                assert kee not in d
                d[kee] = positional_arg

            # OUTPUTS
            _fun = ast.parse(inspect.getsource(f)).body[0]
            _kees = [i.s for i in [i for i in _fun.body if type(i) == ast.Return][0].value.keys]
            assert len(_kees) > 0

            with Experiment(xp_name) as ex:
                input_lits = []
                output_lits = []
                for kee in d:
                    if util.isIterable(d[kee]):
                        input_lits.append(ex.literalForEach(d[kee], kee))
                    else:
                        input_lits.append(ex.literal(d[kee], kee))
                do_action = ex.action(tru_f, input_lits)
                for kee in _kees:
                    output_lits.append(ex.literal(name=kee, parent=do_action))

            pullable_lit = output_lits[0]
            plot = pullable_lit.plot()
            pullable_lit.pull()
            return plot

        return callable_function

    return function_processor
