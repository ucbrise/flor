#!/usr/bin/env python3

import inspect

import ast

from flor import util
from flor import global_state
from sklearn.externals import joblib

from flor.experiment import Experiment


def func(foo):
    """
    Function decorator
    :param foo: Function
    """
    if global_state.interactive:
        if global_state.nb_name is None:
            raise ValueError("Please call flor.setNotebookName")
        filename = inspect.getsourcefile(foo).split('/')[-1]
        if '.py' not in filename[-3:]:
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


def deprecated_func(lambdah):
    if global_state.interactive:
        if global_state.nb_name is None:
            raise ValueError("Please call flor.setNotebookName")
        filename = global_state.nb_name
    else:
        filename = inspect.getsourcefile(lambdah).split('/')[-1]

    def wrapped_func(in_artifacts, out_artifacts):
        if in_artifacts:
            in_args = []
            for in_art in [in_art.loc if util.isFlorClass(in_art) else in_art for in_art in in_artifacts]:
                if util.isPickle(in_art):
                    try:
                        x = util.unpickle(in_art)
                    except:
                        x = joblib.load(in_art)
                elif util.isCsv(in_art):
                    x = in_art
                elif util.isLoc(in_art):
                    with open(in_art, 'r') as f:
                        x = [i.strip() for i in f.readlines() if i.strip()]
                    if len(x) == 1:
                        x = x[0]
                else:
                    x = in_art
                in_args.append(x)
            outs = lambdah(*in_args)
        else:
            outs = lambdah()
        if util.isIterable(outs):
            try:
                assert len(outs) == len(out_artifacts)
                for out, out_loc in zip(outs, [out_art.loc for out_art in out_artifacts]):
                    if util.isPickle(out_loc):
                        try:
                            util.pickleTo(out, out_loc)
                        except:
                            joblib.dump(out, out_loc)
                    else:
                        with open(out_loc, 'w') as f:
                            if util.isIterable(out):
                                for o in out:
                                    f.write(str(o) + '\n')
                            else:
                                f.write(str(out) + '\n')
            except:
                assert len(out_artifacts) == 1
                outs = [outs, ]
                for out, out_loc in zip(outs, [out_art.loc for out_art in out_artifacts]):
                    if util.isPickle(out_loc):
                        util.pickleTo(out, out_loc)
                    else:
                        with open(out_loc, 'w') as f:
                            if util.isIterable(out):
                                for o in out:
                                    f.write(str(o) + '\n')
                            else:
                                f.write(str(out) + '\n')
        elif out_artifacts and outs is not None:
            out_loc = out_artifacts[0].loc
            if util.isPickle(out_loc):
                util.pickleTo(outs, out_loc)
            else:
                with open(out_loc, 'w') as f:
                    if util.isIterable(outs):
                        for o in outs:
                            f.write(str(o) + '\n')
                    else:
                        f.write(str(outs) + '\n')
        else:
            raise AssertionError("Missing location to write or Missing return value.")
        return lambdah.__name__
    return filename, lambdah.__name__, wrapped_func
