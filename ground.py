import requests, json, numpy as np

"""
Abstract class: do not instantiate
"""
class GroundAPI:
	
	headers = {"Content-type": "application/json"}
	
	### EDGES ###
	def createEdge(self, sourceKey, fromNodeId, toNodeId, name="null"):
		d = {
			"sourceKey": sourceKey,
			"fromNodeId": fromNodeId,
			"toNodeId": toNodeId,
			"name": name
		}
		return d

	def createEdgeVersion(self, edgeId, fromNodeVersionStartId, toNodeVersionStartId):
		d = {
			"edgeId": edgeId,
			"fromNodeVersionStartId": fromNodeVersionStartId,
			"toNodeVersionStartId": toNodeVersionStartId
		}
		return d

	def getEdge(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getEdge")

	def getEdgeLatestVersions(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getEdgeLatestVersions")

	def getEdgeHistory(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getEdgeHistory")

	def getEdgeVersion(self, edgeId):
		raise NotImplementedError("Invalid call to GroundClient.getEdgeVersion")

	### NODES ###
	def createNode(self, sourceKey, name="null"):
		d = {
			"sourceKey": sourceKey,
			"name": name
		}
		return d

	def createNodeVersion(self, nodeId, tags=None, parentIds=None):
		d = {
			"nodeId": nodeId
		}
		if tags is not None:
			d["tags"] = tags
		if parentIds is not None:
			d["parentIds"] = parentIds
		return d

	def getNode(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getNode")

	def getNodeLatestVersions(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getNodeLatestVersions")

	def getNodeHistory(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getNodeHistory")

	def getNodeVersion(self, nodeId):
		raise NotImplementedError("Invalid call to GroundClient.getNodeVersion")

	def getNodeVersionAdjacentLineage(self, nodeid):
		raise NotImplementedError("Invalid call to GroundClient.getNodeVersionAdjacentLineage")
	
	### GRAPHS ###
	def createGraph(self, sourceKey, name="null"):
		d = {
			"sourceKey": sourceKey,
			"name": name
		}
		return d

	def createGraphVersion(self, graphId, edgeVersionIds):
		d = {
			"graphId": graphId,
			"edgeVersionIds": edgeVersionIds
		}
		return d

	def getGraph(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getGraph")

	def getGraphLatestVersions(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getGraphLatestVersions")

	def getGraphHitory(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getGraphHitory")

	def getGraphVersion(self, graphId):
		raise NotImplementedError("Invalid call to GroundClient.getGraphVersionh")

class GitImplementation(GroundAPI):

	def __init__(self):
		pass

		### EDGES ###
	def createEdge(self, sourceKey, fromNodeId, toNodeId, name="null"):
		pass

	def createEdgeVersion(self, edgeId, fromNodeVersionStartId, toNodeVersionStartId):
		pass

	def getEdge(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getEdge")

	def getEdgeLatestVersions(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getEdgeLatestVersions")

	def getEdgeHistory(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getEdgeHistory")

	def getEdgeVersion(self, edgeId):
		raise NotImplementedError("Invalid call to GroundClient.getEdgeVersion")

	### NODES ###
	def createNode(self, sourceKey, name="null"):
		pass

	def createNodeVersion(self, nodeId, tags=None, parentIds=None):
		pass

	def getNode(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getNode")

	def getNodeLatestVersions(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getNodeLatestVersions")

	def getNodeHistory(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getNodeHistory")

	def getNodeVersion(self, nodeId):
		raise NotImplementedError("Invalid call to GroundClient.getNodeVersion")

	def getNodeVersionAdjacentLineage(self, nodeid):
		raise NotImplementedError("Invalid call to GroundClient.getNodeVersionAdjacentLineage")
	
	### GRAPHS ###
	def createGraph(self, sourceKey, name="null"):
		pass

	def createGraphVersion(self, graphId, edgeVersionIds):
		pass

	def getGraph(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getGraph")

	def getGraphLatestVersions(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getGraphLatestVersions")

	def getGraphHitory(self, sourceKey):
		raise NotImplementedError("Invalid call to GroundClient.getGraphHitory")

	def getGraphVersion(self, graphId):
		raise NotImplementedError("Invalid call to GroundClient.getGraphVersionh")

class GroundImplementation(GroundAPI):

	def __init__(self, host='localhost', port=9000):
		self.host = host
		self.port = str(port)
		self.url = "http://" + self.host + ":" + self.port

class GroundClient():

	def __new__(*args, **kwargs):
		if args and args[1].strip().lower() == 'git':
			return GitImplementation(**kwargs)
		elif args and args[1].strip().lower() == 'ground':
			# EXAMPLE CALL: GroundClient('ground', host='localhost', port=9000)
			return GroundImplementation(**kwargs)
		else:
			raise ValueError("Backend not supported. Please choose 'git' or 'ground'")