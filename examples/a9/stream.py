#!/usr/bin/env python3
import flor

with flor.Experiment('stream') as ex:
    @flor.func
    def repeat(data, order, dest, **kwargs):
        with open(data, 'r') as f:
            with open(dest, 'w') as o:
                for line in f:
                    o.write("{}: {}\n".format(order, line))
                    o.write("{}: {}\n".format(order, line))

    order = ex.literalForEach(["first", "second"], "order")
    data = ex.artifact('data.txt', 'data')

    do_repeat = ex.action(repeat, [data, order])
    dest = ex.artifact('dest.txt', 'dest', do_repeat)

dest.plot()
dest.pull()

