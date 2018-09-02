#!/usr/bin/env python3
import flor

with flor.Experiment('plate_demo') as ex:
    @flor.func
    def double(ones, **kwargs):
        return {'twice': 2*ones}

    @flor.func
    def triple(tens, **kwargs):
        return {'thrice': 3*tens}

    @flor.func
    def multiply(twice, thrice, **kwargs):
        z = twice * thrice
        print(z)
        return {'product': z}


    ones = ex.literalForEach(v=[1,2,3], name="ones", default=2)
    tens = ex.literalForEach(v=[10, 100], name="tens", default=10)

    doDouble = ex.action(double, [ones, ])
    twice = ex.literal(name="twice", parent=doDouble)

    doTriple = ex.action(triple, [tens])
    thrice = ex.literal(name="thrice", parent=doTriple)

    doMultiply = ex.action(multiply, [twice, thrice])
    product = ex.literal(name="product", parent=doMultiply)

product.plot()
product.pull()

