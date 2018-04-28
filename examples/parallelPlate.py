import flor

with flor.Experiment('plate_demo') as ex:
	ex.groundClient('ground')

	ones = ex.literal([1, 2, 3], "ones")
	ones.forEach()

	tens = ex.literal([10, 100], "tens")
	tens.forEach()

	@flor.func
	def multiply(x, y):
	    z = x*y
	    print(z)
	    return z

	doMultiply = ex.action(multiply, [ones, tens])
	product = ex.artifact('product.txt', doMultiply)

product.parallelPull()
# product.plot()
