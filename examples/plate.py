#!/usr/bin/env python3
import project

project.groundClient('git')
project.jarvisFile('plate.py')

ones = project.Literal([1, 2, 3], "ones")
ones.forEach()

tens = project.Literal([10, 100], "tens")
tens.forEach()

@project.func
def multiply(x, y):
    z = x*y
    print(z)
    return z

doMultiply = project.Action(multiply, [ones, tens])
product = project.Artifact('product.txt', doMultiply)

# product.pull()
# product.plot()
product_df = product.parallelPull(manifest=True)
project.Util.pickleTo(product_df, 'product.pkl')