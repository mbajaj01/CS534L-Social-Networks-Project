import os
import sys
import heapq
import numpy as np
import Queue
import networkx as nx
from classDefinitions import *
from scipy.misc import logsumexp, comb
import math
from cStringIO import StringIO
import cPickle

def generateGraph(nodes, density=0.5):
	G = nx.DiGraph()
	G.add_nodes_from(range(nodes))
	edgeProbs = np.random.rand(nodes,nodes)
	edgeOccurence = edgeProbs < density
	edgeIndices = np.transpose(edgeOccurence.nonzero())
	for id,edge in enumerate(edgeIndices):
		node1 = edge[0]
		node2 = edge[1]
		probabilities = np.random.random()
		G.add_edges_from([(node1, node2, {'probabilities': probabilities, 'estimates': 0., 'average': 0})])
	return G

class IC(object):
	def __init__(self, graph):
		self.graph = graph
		self.lam = None

	def generatePossibleWorld(self):
		coinFlips = np.random.rand(self.graph.number_of_nodes(), self.graph.number_of_nodes())
		return coinFlips

	def initAllItems(self):
		nx.set_node_attributes(self.graph, {n: False for n in self.graph.nodes}, 'isInfluenced')
		nx.set_node_attributes(self.graph, {n: None for n in self.graph.nodes}, 'influenceTimestep')
		nx.set_node_attributes(self.graph, {n: None for n in self.graph.nodes}, 'influencedBy')
		nx.set_node_attributes(self.graph, {n: {} for n in self.graph.nodes}, 'influencedNeighbours')
		nx.set_node_attributes(self.graph, {n: [] for n in self.graph.nodes}, 'influenced')
		nx.set_edge_attributes(self.graph, {e: False for e in self.graph.edges}, 'isLive')


	def diffusion(self, seeds, isEnvironment, possibleWorld=None, returnPossibleWorld=False):
		if possibleWorld is None:
			#possibleWorld = self.generatePossibleWorld()
			world = {}
		else:
			world = possibleWorld
		activeNodes = Queue.Queue()
		for seed in seeds:
			activeNodes.put([seed, 0])
			self.graph.node[seed]['isInfluenced'] = True
			self.graph.node[seed]['influencedBy'] = -1
			self.graph.node[seed]['influenceTimestep'] = 0
		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			world[currentNode] = {}
			for node in self.graph.successors(currentNode):
				if not self.graph.node[node]['isInfluenced']:
					self.graph.node[node]['influencedNeighbours'][currentNode] = timestep
					if isEnvironment:
						# dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
						dot = self.graph.edges[currentNode, node]['probabilities']
					else:
						# dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
						dot = self.graph.edges[currentNode, node]['estimates']
					if possibleWorld is None:
						world[currentNode][node] = np.random.random()
					prob = 0.
					try:
						success = dot > world[currentNode][node]
					except:
						success = False
					if success:
						activeNodes.put([node, timestep+1])
						self.graph.node[node]['isInfluenced'] = True
						self.graph.node[node]['influencedBy'] = currentNode
						self.graph.node[node]['influenced'] = []
						self.graph.node[currentNode]['influenced'].append(node)
						self.graph.node[node]['influenceTimestep'] = timestep + 1
						self.graph.edges[currentNode, node]['isLive'] = True
				else:
					if timestep < self.graph.node[node]['influenceTimestep']:
						self.graph.node[node]['influencedNeighbours'][currentNode] = timestep

		influencedNodes = [n for n in self.graph.nodes if self.graph.node[n]['isInfluenced'] == True]
		#print len(influencedNodes)
		if returnPossibleWorld:
			return len(influencedNodes), possibleWorld
		else:
			return len(influencedNodes)

	def expectedSpread(self, seeds, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		for simulation in range(numberOfSimulations):
			self.initAllItems()
			expSpread += self.diffusion(seeds, isEnvironment)
		expSpread = expSpread/numberOfSimulations
		return expSpread

	def findBestSeeds(self, budget, isEnvironment=True, numberOfSimulations=100):
		seeds = []
		seedMap = {}
		while len(seeds) < budget:
			maximumSpread = 0
			newSeed = -1
			for candidate in range(self.graph.number_of_nodes()):
				try:
					isSeed = seedMap[candidate]
				except:
					expSpread = self.expectedSpread(seeds+[candidate],  numberOfSimulations, isEnvironment)
					if expSpread > maximumSpread:
						maximumSpread = expSpread
						newSeed = candidate
			if newSeed != -1:
				seeds.append(newSeed)
				seedMap[newSeed] = 1
			else:
				break
		return seeds, maximumSpread

	def RRGreedy(self, budget, isEnvironment=True,  l=1., epsilon=0.2):
		self.R = nx.Graph()
		# KPT = self.KPTEstimation(budget, isEnvironment, l, epsilon)
		# n = self.graph.number_of_nodes()
		# if self.lam == None:
		# 	self.lam = (8+2*epsilon)*n*(l*np.log(n) + np.log(comb(n,budget)) + np.log(2))*(epsilon**-2)
		# theta = int(np.ceil(self.lam/KPT))
		theta = 1000
		for r in range(theta):
			RRset = self.generateRRSet(isEnvironment)
		seeds = []
		for i in range(budget):
			maxdegree = 0
			maxnode = 0
			for node in self.R.nodes:
				degree = self.R.degree(node)
				if degree > maxdegree:
					maxnode = node
					maxdegree = degree
			try:
				self.R.remove_node(maxnode)
				seeds.append(maxnode)
			except:
				maxnode = np.random.choice(self.graph.nodes, 1)
				seeds.append(maxnode[0])
		return seeds
	
	def KPTEstimation(self, budget, isEnvironment, l, epsilon):
		n = self.graph.number_of_nodes()
		ci = (6*l*np.log(n) + 6*np.log(np.log2(n)))
		for i in range(1, int(np.floor(np.log2(n)-1))):
			c = ci*(2**i)
			total = 0.
			for j in range(1, int(np.floor(c))):
				wr = self.generateRRSet(isEnvironment)
				kr = 1 - (1 - (wr/self.graph.number_of_edges()))**budget
				total += kr
			if (total/c) > (1/(2**i)):
				return (n*total)/(2*c)
		return 1



	def generateRRSet(self, isEnvironment=True):
		randomNode = np.random.choice(self.graph.nodes, 1)
		activeNodes = Queue.Queue()
		activeNodes.put([randomNode[0],0])
		isProcessed = {}
		isProcessed[randomNode[0]] = 1
		wr = 0.
		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			for node in self.graph.predecessors(currentNode):
				if node not in isProcessed:
					if isEnvironment:
						dot = self.graph.edges[node, currentNode]['probabilities']
					else:
						dot = self.graph.edges[node, currentNode]['estimates']
					if dot > np.random.random():
						activeNodes.put([node, timestep+1])
						wr += len(list(self.graph.predecessors(node)))
						isProcessed[node] = 1
						self.R.add_edge(node, currentNode)
		return wr


class MAB(object):
	def __init__(self, tic, budget):
		self.budget = budget
		self.tic = tic

	def explore(self):
		seeds = np.random.choice(list(self.tic.graph.nodes), self.budget, replace=False)
		return seeds.tolist()

	def exploit(self):
		seeds = self.tic.celf(self.budget, isEnvironment=False)
		return seeds

	def epsilonGreedy(self, epsilon):
		if epsilon > np.random.rand(1):
			seeds = self.explore()
		else:
			seeds = self.exploit()
		spread = self.tic.diffusion(seeds=seeds, isEnvironment=True)
		return seeds

	def logLikelihood(self, pi):
		likelihood = 0.
		for i in range(self.tic.numItems):				
			cascadeLogProbPositive = np.zeros(self.tic.numTopics)
			cascadeLogProbNegative = np.zeros(self.tic.numTopics)
			for node1 in self.tic.graph.nodes:
				for node2 in self.tic.graph.node[node1]['influencedNeighbours'][i].keys():
					if node2 == self.tic.graph.node[node1]['influencedBy'][i]:
						mask = self.tic.graph.edges[node2, node1]['estimates'] == 0.
						cascadeLogProbPositive += np.log(self.tic.graph.edges[node2, node1]['estimates'] + mask*1.)
						self.tic.graph.edges[node2, node1]['totalActivations'][i] += 1
					else:
						mask = self.tic.graph.edges[node2, node1]['estimates'] == 1.
						cascadeLogProbNegative += np.log((1. - self.tic.graph.edges[node2, node1]['estimates']) + mask*1.)
			likelihood += np.sum(self.tic.items[i].topicEstimates*(pi + cascadeLogProbPositive + cascadeLogProbNegative))
		return likelihood

	def L2Error(self):
		l2 = 0.
		for edge in self.tic.graph.edges:
			l2 += (self.tic.graph.edges[edge]['estimates'] - self.tic.graph.edges[edge]['probabilities'])**2
		return l2

	def learner(self, iterations, epsilon):
		edgeActivations = {e:0. for e in self.tic.graph.edges}
		nodeActivations = {n:0 for n in self.tic.graph.nodes}
		for t in range(iterations):
			self.tic.initAllItems()
			np.random.seed()
			seeds = self.epsilonGreedy(epsilon)
			for node in self.tic.graph.nodes:
				if self.tic.graph.nodes[node]['isInfluenced']:
					nodeActivations[node] += 1
					if self.tic.graph.nodes[node]['influencedBy'] != -1:
						edgeActivations[(self.tic.graph.nodes[node]['influencedBy'], node)] += 1
						key = (self.tic.graph.nodes[node]['influencedBy'], node)
			for e in self.tic.graph.edges:
				if nodeActivations[e[1]] > 0:
					self.tic.graph.edges[e]['estimates'] = edgeActivations[e]/nodeActivations[e[1]]
			print self.L2Error(), self.tic.graph.edges[key]['estimates'], self.tic.graph.edges[key]['probabilities'], self.tic.graph.number_of_edges()

	def learnerNode(self, iterations, epsilon, initialIter = 0):
		S_w = {}
		positive_sum = {}
		seedsTrue = self.tic.RRGreedy(5, isEnvironment=True)
		# expSpreadTrue = self.tic.expectedSpread(seedsTrue, 10, True)
		regret = 0.
		seedsBandit = self.tic.RRGreedy(5, isEnvironment=False)
		expSpreadTrue = 0.
		expSpreadBandit = 0.
		for diffIter in range(1):
			#possibleWorld = self.tic.generatePossibleWorld()
			self.tic.initAllItems()
			expSpreadTrue, possibleWorld = self.tic.diffusion(seedsTrue, isEnvironment=True, returnPossibleWorld=True)
			self.tic.initAllItems()
			expSpreadBandit = self.tic.diffusion(seedsBandit, isEnvironment=True, possibleWorld=possibleWorld)
		expSpreadTrue /= 1
		expSpreadBandit /= 1
		print expSpreadTrue, expSpreadBandit
		regret += ( expSpreadTrue - expSpreadBandit)
		print self.L2Error(), regret
		for t in range(iterations):
			self.tic.initAllItems()
			np.random.seed()
			seeds = self.epsilonGreedy(epsilon)
			P_w = {}
			isPostiveParent = {}
			for node in self.tic.graph.nodes:
				prob = 1.
				for parent in self.tic.graph.node[node]['influencedNeighbours'].keys():
					if (parent,node) not in S_w:
						S_w[(parent, node)] = {'positive':0, 'negative':0}
					isPostiveParent[(parent, node)] = False
					if self.tic.graph.node[node]['influenceTimestep'] == self.tic.graph.node[node]['influencedNeighbours'][parent] + 1:
						prob *= 1. - self.tic.graph.edges[parent, node]['estimates']
						S_w[(parent, node)]['positive'] += 1
						isPostiveParent[(parent, node)] = True
					else:
						S_w[(parent, node)]['negative'] += 1
				if self.tic.graph.node[node]['isInfluenced'] and self.tic.graph.node[node]['influenceTimestep'] > 0:
					P_w[node] = 1. - prob
					if P_w[node] == 0.:
						P_w[node] = 1.
				
			for edge in self.tic.graph.edges:
				if edge in isPostiveParent:
					if edge in S_w:
						denominator = S_w[edge]['negative'] + S_w[edge]['positive']
						if edge not in positive_sum:
							positive_sum[edge] = 0.
						numerator = positive_sum[edge]
						if isPostiveParent[edge]:
							numerator +=  self.tic.graph.edges[edge]['estimates']/P_w[edge[1]]
						if numerator == 0.:
							if S_w[edge]['positive'] > 0:
								numerator = (S_w[edge]['positive'])/(S_w[edge]['negative'] + S_w[edge]['positive'])
						positive_sum[edge] = numerator
						if t > initialIter:
							self.tic.graph.edges[edge]['estimates'] =  numerator/denominator
						key = edge
			if t%1 == 0 and t>=initialIter:
				seedsBandit = self.tic.RRGreedy(5, isEnvironment=False)
				expSpreadTrue = 0.
				expSpreadBandit = 0.
				for diffIter in range(1):
					#possibleWorld = self.tic.generatePossibleWorld()
					self.tic.initAllItems()
					expSpreadTrue, possibleWorld = self.tic.diffusion(seedsTrue, isEnvironment=True, returnPossibleWorld=True)
					self.tic.initAllItems()
					expSpreadBandit = self.tic.diffusion(seedsBandit, isEnvironment=True, possibleWorld=possibleWorld)
				expSpreadTrue /= 1
				expSpreadBandit /= 1
				print expSpreadTrue, expSpreadBandit
				regret += ( expSpreadTrue - expSpreadBandit)
				print self.L2Error(), regret/(t+2)

		
itemList = []
numItems = 5
numTopics = 3
budget = 50
# for i in range(numItems):
# 	itemDist = np.random.rand(numTopics)
# 	itemList.append(Item(i, itemDist/sum(itemDist)))
graph = cPickle.load(open('data/wikiGraph.p'))
# nx.set_edge_attributes(graph, {e: np.random.random() for e in graph.edges}, 'estimates')
# nx.set_edge_attributes(graph, {e: np.random.random() for e in graph.edges}, 'probabilities')
print graph.number_of_edges(), graph.number_of_nodes()
tic = IC(graph)
# tic.RRGreedy(budget)
mab = MAB(tic, budget)
print mab.learnerNode(100000, 1)