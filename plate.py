#!/usr/bin/env python3
import jarvis

jarvis.ground_client('git')
jarvis.jarvisFile('plate.py')

ith_param = jarvis.ForEach(jarvis.Sample(1.0, 'params.txt'))

ith_param2 = jarvis.ForEach(jarvis.Sample(1.0, 'params2.txt'))

from mult import multiply
do_multiply = jarvis.Action(multiply, [ith_param, ith_param2])
product = jarvis.Artifact('product.txt', do_multiply)

ith_param3 = jarvis.Fork(ith_param2)

do_multiply = jarvis.Action(multiply, [product, ith_param3])
product = jarvis.Artifact('product2.txt', do_multiply)

product.pull()
product.plot()