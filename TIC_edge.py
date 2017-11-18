import os
import sys
import snap
import numpy as np
import Queue

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
		self.estimates = np.zeros(len(self.probabilities))

class Graph(object):
	def __init__(self, nodes, edges):
		self.nodes = nodes
		self.edges = edges


class Item(object):
	def __init__(self, id, topicDistribution):
		self.id = id
		self.topicDistribution = topicDistribution
		self.topicEstimates = np.zeros(len(self.topicDistribution))

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
	def __init__(self, graph, numTopics, numItems):
		self.graph = graph
		self.numTopics = numTopics
		self.numItems = numItems
		

	def diffusion(self, seeds, item, isEnvironment, returnCascade=False):
		possibleWorld = self.generatePossibleWorld()
		influencedNodes = seeds[:]
		activeNodes = Queue.Queue()
		influencedMap = {x:0 for x in range(len(self.graph.nodes))}
		activeEdges = []
		for seed in seeds:
			activeNodes.put([seed,0])
			influencedMap[seed] = 1
		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			for edge, node in self.graph.nodes[currentNode].outEdges.items():
				if influencedMap[node] == 0:
					if isEnvironment:
						dot = np.dot(self.graph.edges[edge].probabilities, item)
					else:
						np.dot(self.graph.edges[edge].estimates, item)				
					if dot > possibleWorld[edge]:
						activeNodes.put([node, timestep+1])
						influencedNodes.append(node)
						influencedMap[node] = 1
						activeEdges.append(edge)
		if returnCascade:
			return len(influencedNodes), activeEdges
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
		coinFlips = np.random.rand(len(graph.edges))
		return coinFlips


class MAB(object):
	def __init__(self, tic, budget):
		self.budget = budget
		self.tic = tic

	def explore(self, item):
		seeds = np.random.choice(len(self.tic.graph.nodes), self.budget, replace=False)
		return seeds

	def exploit(self, item):
		seeds,_ = self.tic.findBestSeeds(item, self.budget, isEnvironment=False)
		return seeds

	def epsilonGreedy(self, epsilon, item):
		if epsilon > np.random.rand(1):
			seeds = self.explore(item)
		else:
			seeds = self.exploit(item)
		spread, activeEdges = self.tic.diffusion(self, seeds=seeds, item=item, isEnvironment=True, returnCascade=True)
		return activeEdges

	def learner(self, iterations, epsilon, item):

		for t in range(iterations):
			activeEdges = self.epsilonGreedy(epsilon, item)




graph = generateGraph(100,2, density=0.2)
tic = TIC(graph, 2)
print tic.findBestSeeds(np.array([0.5, 0.5]), 3)