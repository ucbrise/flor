import flor

ex = flor.Experiment('plate_demo')

ex.groundClient('git')

#your commit hash experience may vary
temp = flor.fork('plate_demo', '1531910097076910998f5626da0f3fe84e9bd113', "temp", ex.xp_state)

# ones = ex.literal([1, 2], "ones")
# ones.forEach()

# tens = ex.literal([10, 100], "tens")
# tens.forEach()

# @flor.func
# def multiply(x, y):
#     z = x*y
#     print(z)
#     return z

# doMultiply = ex.action(multiply, [ones, tens])
# product = ex.artifact('product.txt', doMultiply)

# product.parallelPull()
# product.plot()
