#!/usr/bin/env python3

import inspect

from . import util
from . import global_state


def func(lambdah):
    if global_state.interactive:
        if global_state.nb_name is None:
            raise ValueError("Please call jarvis.setNotebookName")
        filename = global_state.nb_name
    else:
        filename = inspect.getsourcefile(lambdah).split('/')[-1]

    def wrapped_func(in_artifacts, out_artifacts):
        if in_artifacts:
            in_args = []
            for in_art in [in_art.loc if util.isJarvisClass(in_art) else in_art for in_art in in_artifacts]:
                if util.isPickle(in_art):
                    x = util.unpickle(in_art)
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
                        util.pickleTo(out, out_loc)
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
