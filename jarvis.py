#!/usr/bin/env python3

class Precondition:

	def __init__(self):
		pass

	def check(self):
		pass

class Postcondition:

	def __init__(self):
		pass

	def check(self):
		pass

class Task:

	def __init__(self, func, preconditions, postconditions):
		self.func = func
		self.preconditions = preconditions
		self.postconditions = postconditions

	def run(self):
		for precondition in self.preconditions:
			precondition.check()
		result = self.func()
		for postcondition in self.postconditions:
			postcondition.check()
		return result

class Metadata(Precondition):

	def __init__(self):
		pass

class Data(Precondition):

	def __init__(self):
		pass

class Model(Precondition):

	def __init__(self):
		pass

class Test(Postcondition):

	def __init__(self):
		pass