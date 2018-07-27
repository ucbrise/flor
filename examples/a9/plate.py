#!/usr/bin/env python3
import flor

with flor.Experiment('plate_demo') as ex:
    @flor.func
    def double(x):
        return 2*x

    @flor.func
    def triple(x):
        return 3*x

    @flor.func
    def multiply(x, y):
        z = x*y
        print(z)
        return z


    ones = ex.literalForEach(v=[i for i in range(1000)], name="ones", default=2)
    tens = ex.literalForEach(v=[i for i in range(1000)], name="tens", default=10)

    doDouble = ex.action(double, [ones, ])
    twice = ex.literal(name="twice", parent=doDouble)

    doTriple = ex.action(triple, [tens])
    thrice = ex.literal(name="thrice", parent=doTriple)

    doMultiply = ex.action(multiply, [twice, thrice])
    product = ex.literal(name="product", parent=doMultiply)

# product.plot()
product.pull()

