#!/usr/bin/env python3
import os, sys, git

class Artifact:

	# loc: location
	# typ: type
	# parent: each artifact is produced by 1 action
	def __init__(self, loc, typ, parent):
		self.loc = loc
		self.dir = self.loc.split('.')[0] + ".d"
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
		# Now the artifact exists, do git
		# We resolve the directory name by loc
		dir_name = self.dir
		# If the directory not exists, need to init repo
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
			# Move new file to its repo
			os.rename(self.loc, dir_name + "/" + self.loc)
			os.chdir(dir_name)
			repo = git.Repo.init(os.getcwd())
			repo.index.add([self.loc])
			repo.index.commit("initial commit")
			os.chdir('../')
		else:
			os.rename(self.loc, dir_name + "/" + self.loc)
			os.chdir(dir_name)
			repo = git.Repo(os.getcwd())
			repo.index.add([self.loc])
			repo.index.commit("incremental commit")
			os.chdir('../')

	"""
	Specify the intent:
	r -> read
	w -> write
	Are you getting the location to read or write?
	This is a workaround for how git artifact versioning is implemented
	"""
	def getLocation(self, intent):
		if intent == 'r':
			return self.dir + "/" + self.loc
		elif intent == 'w':
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