#!/usr/bin/env python3
import flor

ex = flor.Experiment('plate_demo')

ex.groundClient('ground')

ones = ex.literalForEach([1, 2, 3], "ones", default=3)
#ones.forEach()

tens = ex.literalForEach([10, 100], "tens", default=10)
#tens.forEach()

@flor.func
def multiply(x, y):
    z = x*y
    print(z)
    return z

doMultiply = ex.action(multiply, [ones, tens])
product = ex.artifact('product.txt', doMultiply)

product.plot()
#product.pull()
product.peek(bindings = {ones: 2, tens: 100})
