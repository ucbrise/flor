#!/usr/bin/env python3
import jarvis

jarvis.ground_client('git')
jarvis.jarvisFile('plate.py')

ith_param = jarvis.Sample(1.0, 'params.txt', False, 3)

ith_param2 = jarvis.Sample(1.0, 'params2.txt', False)

from mult import multiply
do_multiply = jarvis.Action(multiply, [ith_param, ith_param2])
product = jarvis.Artifact('product.txt', do_multiply)

product.pull()
product.plot()