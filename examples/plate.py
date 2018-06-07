#!/usr/bin/env python3
import flor

with flor.Experiment('plate_demo') as ex:

	ex.groundClient('ground')
  ones = ex.literalForEach([1, 2, 3], "ones", default=3)

  tens = ex.literalForEach([10, 100], "tens", default=10)

	@flor.func
	def multiply(x, y):
	    z = x*y
	    print(z)
	    return z

	doMultiply = ex.action(multiply, [ones, tens])
	product = ex.artifact('product.txt', doMultiply)

  product.plot()
  product.peek(bindings = {ones: 2, tens:100})
