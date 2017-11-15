#!/usr/bin/env python3
import jarvis

jarvis.ground_client('git')
jarvis.jarvisFile('plate.py')

ith_param = jarvis.Sample(1.0, 'params.txt', False)

ith_param2 = jarvis.Sample(1.0, 'params2.txt', False)

from mult import multiply
do_multiply = jarvis.Action(multiply, [ith_param, ith_param2])
product = jarvis.Artifact('product.txt', do_multiply)

ith_param3 = jarvis.Sample(1.0, 'params3.txt', False)

do_multiply2 = jarvis.Action(multiply, [product, ith_param2])
product2 = jarvis.Artifact('product2.txt', do_multiply2)

product2.pull()
product2.plot()

ith_param3.pull()
ith_param3.plot()