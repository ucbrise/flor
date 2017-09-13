#!/usr/bin/env python3
import os, sys

class Artifact:

	# loc: location
	# typ: type
	# parent: each artifact is produced by 1 action
	def __init__(self, loc, typ, parent):
		self.loc = loc
		self.typ = typ
		self.parent = parent
		# Need a way to manage versions, possibly with Ground integration
		self.version = None

		# Now we bind the artifact to its parent
		self.parent.out_artifacts.append(self)
		self.parent.out_types.append(typ)


	def pull(self):
		print(self.loc)
		self.parent.run()

	def getLocation(self):
		return self.loc


	def hasChanged(self):
		pass

	"""
	We will want to check the loc prefix to decide where to look
	for existence e.g. http, s3, sftp, etc.
	No pre-defined prefix, assume local filesystem.
	"""
	def exists(self):
		if not os.path.isfile(self.loc):
			print(self.loc + " not found.")
			sys.exit(1)

	"""
	We assume an open-ended Integrity predicate on each artifact. 
	The design of this predicate language is TBD
	"""
	def isLegal(self):
		pass

	def stat(self):
		pass


class Action:

	def __init__(self, func, in_artifacts=None):
		self.func = func
		self.out_artifacts = []
		self.out_types = []
		self.in_artifacts = in_artifacts


	def run(self):
		if self.in_artifacts:
			for artifact in self.in_artifacts:
				artifact.pull()
		self.func(self.in_artifacts, self.out_artifacts, self.out_types)

	def produce(self, loc, typ):
		return Artifact(loc, typ, self)

		

__valid_types__ = {"metadata", "data", "model", "script"}