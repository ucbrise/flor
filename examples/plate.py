#!/usr/bin/env python3
import flor

with flor.Experiment('plate_demo') as ex:

    ones = ex.literalForEach(v=[1, 2, 3], name="ones", default=3)

    tens = ex.literalForEach(v=[10, 100], name="tens", default=10)

    @flor.func
    def multiply(x, y):
        z = x*y
        print(z)
        return z

    doMultiply = ex.action(multiply, [ones, tens])
    product = ex.literal(name="product", parent=doMultiply)

product.plot()

