import os
import sys
import numpy as np

class Node(object):
	def __init__(self, id, inEdges, outEdges):
		self.id = id
		self.inEdges = inEdges
		self.outEdges = outEdges

class Edge(object):
	def __init__(self, id, node1, node2, probabilities):
		self.id = id
		self.node1 = node1
		self.node2 = node2
		self.probabilities = probabilities
		self.estimates = np.random.rand(len(self.probabilities))

class Graph(object):
	def __init__(self, nodes, edges):
		self.nodes = nodes
		self.edges = edges


class Item(object):
	def __init__(self, id, topicDistribution):
		self.id = id
		self.topicDistribution = topicDistribution
		self.topicEstimates = np.random.rand(len(self.topicDistribution))
		self.topicEstimates = self.topicEstimates/sum(self.topicEstimates)
		self.topicAverage = np.zeros(len(self.topicDistribution))

class CascadeLog(object):
	def __init__(self, node1, node2, edge, time):
		self.node1 = node1
		self.node2 = node2
		self.edge = edge
		self.time = time 

class NodeCascade(object):
	def _init__(self, node, time):
		self.node = node
		self.time = time
