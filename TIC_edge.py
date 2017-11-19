import os
import sys
import snap
import numpy as np
import Queue
from scipy.misc import logsumexp

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
		self.topicEstimates = np.zeros(len(self.topicDistribution))

class CascadeLog(object):
	def __init__(self, node1, node2, edge, time):
		self.node1 = node1
		self.node2 = node2
		self.edge = edge
		self.time = time

def generateGraph(nodes, topics, density=0.5):
	graph = Graph([],[])
	for node in range(nodes):
		graph.nodes.append(Node(node, {}, {}))

	edgeProbs = np.random.rand(nodes,nodes)
	edgeOccurence = edgeProbs < density
	edgeIndices = np.transpose(edgeOccurence.nonzero())
	for id,edge in enumerate(edgeIndices):
		node1 = edge[0]
		node2 = edge[1]
		probabilities = np.random.rand(topics)
		edge = Edge(id,node1,node2,probabilities)
		graph.edges.append(edge)
		graph.nodes[node1].outEdges[id] = node2
		graph.nodes[node2].inEdges[id] = node1
	return graph

class TIC(object):
	def __init__(self, graph, numTopics, items):
		self.graph = graph
		self.numTopics = numTopics
		self.numItems = len(items)
		self.items = items
		

	def diffusion(self, seeds, item, isEnvironment, returnCascade=False):
		possibleWorld = self.generatePossibleWorld()
		influencedNodes = seeds[:]
		activeNodes = Queue.Queue()
		influencedMap = {x:0 for x in range(len(self.graph.nodes))}
		edgeCascade = {x:0 for x in range(len(self.graph.edges))}
		for seed in seeds:
			activeNodes.put([seed,0])
			influencedMap[seed] = 1
		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			for edge, node in self.graph.nodes[currentNode].outEdges.items():
				if influencedMap[node] == 0:
					if isEnvironment:
						dot = np.dot(self.graph.edges[edge].probabilities, item.topicDistribution)
					else:
						dot = np.dot(self.graph.edges[edge].estimates, item.topicEstimates)				
					if dot > possibleWorld[edge]:
						activeNodes.put([node, timestep+1])
						influencedNodes.append(node)
						influencedMap[node] = 1
						edgeCascade[edge] = 1
		if returnCascade:
			return len(influencedNodes), edgeCascade
		else:
			return len(influencedNodes)

	def expectedSpread(self, seeds, item, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		for simulation in range(numberOfSimulations):
			expSpread += self.diffusion(seeds, item, isEnvironment)
		expSpread = expSpread/numberOfSimulations
		return expSpread

	def findBestSeeds(self, item, budget, isEnvironment=True, numberOfSimulations=100):
		seeds = []
		seedMap = {}
		while len(seeds) < budget:
			maximumSpread = 0
			newSeed = -1
			for candidate in range(len(self.graph.nodes)):
				try:
					isSeed = seedMap[candidate]
				except:
					expSpread = self.expectedSpread(seeds+[candidate], item, numberOfSimulations, isEnvironment)
					if expSpread > maximumSpread:
						maximumSpread = expSpread
						newSeed = candidate
			if newSeed != -1:
				seeds.append(newSeed)
				seedMap[newSeed] = 1
			else:
				break
		return seeds, maximumSpread

	def generatePossibleWorld(self):
		coinFlips = np.random.rand(len(self.graph.edges))
		return coinFlips


class MAB(object):
	def __init__(self, tic, budget):
		self.budget = budget
		self.tic = tic

	def explore(self, item):
		seeds = np.random.choice(len(self.tic.graph.nodes), self.budget, replace=False)
		return seeds.tolist()

	def exploit(self, item):
		seeds,_ = self.tic.findBestSeeds(item, self.budget, isEnvironment=False)
		return seeds

	def epsilonGreedy(self, epsilon, item):
		if epsilon > np.random.rand(1):
			seeds = self.explore(item)
		else:
			seeds = self.exploit(item)
		spread, cascade = self.tic.diffusion(seeds=seeds, item=item, isEnvironment=True, returnCascade=True)
		return cascade

	def learner(self, iterations, epsilon):
		pi = np.random.rand(numTopics)
		pi = np.log(pi/sum(pi))
		edgeActivation = {x:{i:0.0 for i in range(self.tic.numItems)} for x in range(len(self.tic.graph.edges))}
		for t in range(iterations):
			for i in range(self.tic.numItems):
				cascade = self.epsilonGreedy(epsilon, self.tic.items[i])
				cascadeLogProbPositive = np.zeros(self.tic.numTopics)
				cascadeLogProbNegative = np.zeros(self.tic.numTopics)
				for edge, isActive in cascade.items():
					if isActive == 1:
						cascadeLogProbPositive += np.log(self.tic.graph.edges[edge].estimates)
						edgeActivation[edge][i] += 1
					else:
						cascadeLogProbNegative += np.log(1. - self.tic.graph.edges[edge].estimates)
				self.tic.items[i].topicEstimates = (pi + cascadeLogProbPositive + cascadeLogProbNegative)
				self.tic.items[i].topicEstimates = np.exp(self.tic.items[i].topicEstimates - logsumexp(self.tic.items[i].topicEstimates))
			pi = np.sum([self.tic.items[i].topicEstimates for i in range(self.tic.numItems)],0)/self.tic.numItems
			normalizer = pi[:]*self.tic.numItems
			pi = np.log(pi)
			for edge in range(len(self.tic.graph.edges)):
				self.tic.graph.edges[i].estimates = np.sum([(self.tic.items[i].topicEstimates * edgeActivation[edge][i])/(t+1) for i in range(self.tic.numItems)],0)/normalizer




itemList = []
numItems = 2
numTopics = 3
for i in range(numItems):
	itemDist = np.random.rand(numTopics)
	itemList.append(Item(i, itemDist/sum(itemDist)))
tic = TIC(generateGraph(100, numTopics, density=0.1), numTopics, itemList)
mab = MAB(tic, numTopics)
print mab.learner(10, 1)