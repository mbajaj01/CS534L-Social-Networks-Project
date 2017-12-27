import os
import sys
import heapq
import numpy as np
import Queue
import networkx as nx
from classDefinitions import *
from scipy.misc import logsumexp
import math
from cStringIO import StringIO


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
		G.add_edges_from([(node1, node2, {'probabilities': probabilities, 'estimates': np.random.random(), 'average': 0})])
	return G

class IC(object):
	def __init__(self, graph):
		self.graph = graph

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


	def diffusion(self, seeds, isEnvironment):
		possibleWorld = self.generatePossibleWorld()
		activeNodes = Queue.Queue()
		for seed in seeds:
			activeNodes.put([seed, 0])
			self.graph.node[seed]['isInfluenced'] = True
			self.graph.node[seed]['influencedBy'] = -1
			self.graph.node[seed]['influenceTimestep'] = 0
		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			for node in self.graph.successors(currentNode):
				if not self.graph.node[node]['isInfluenced']:
					self.graph.node[node]['influencedNeighbours'][currentNode] = timestep
					if isEnvironment:
						# dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
						dot = self.graph.edges[currentNode, node]['probabilities']
					else:
						# dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
						dot = self.graph.edges[currentNode, node]['estimates']
					if dot > possibleWorld[currentNode][node]:
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
		return len(influencedNodes)

	def diffusionCelf(self, seeds, u, cur_best, isEnvironment):
		possibleWorld = self.generatePossibleWorld()
		activeNodes = Queue.Queue()
		nx.set_node_attributes(self.graph, False, 'isInfluencedSeed')
		nx.set_node_attributes(self.graph, False, 'isInfluencedu')
		nx.set_node_attributes(self.graph, False, 'isInfluencedcur')
		for seed in seeds:
			activeNodes.put([seed, 0])
			self.graph.node[seed]['isInfluencedSeed'] = True

		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			for node in self.graph.successors(currentNode):
				if not self.graph.node[node]['isInfluencedSeed']:
					if isEnvironment:
						# dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
						dot = self.graph.edges[currentNode, node]['probabilities']
					else:
						# dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
						dot = self.graph.edges[currentNode, node]['estimates']
					if dot > possibleWorld[currentNode][node]:
						activeNodes.put([node, timestep+1])
						self.graph.node[node]['isInfluencedSeed'] = True
	
		if not self.graph.node[u]['isInfluencedSeed']:
			activeNodes.put([u, 0])
			self.graph.node[u]['isInfluencedu'] = True
			while not activeNodes.empty():
				currentNode, timestep = activeNodes.get()
				for node in self.graph.successors(currentNode):
					if not self.graph.node[node]['isInfluencedSeed'] and not self.graph.node[node]['isInfluencedu']:
						if isEnvironment:
							# dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
							dot = self.graph.edges[currentNode, node]['probabilities']
						else:
							# dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
							dot = self.graph.edges[currentNode, node]['estimates']
						if dot > possibleWorld[currentNode][node]:
							activeNodes.put([node, timestep+1])
							self.graph.node[node]['isInfluencedu'] = True
		
		if not self.graph.node[cur_best]['isInfluencedSeed']:
			activeNodes.put([cur_best, 0])
			self.graph.node[u]['isInfluencedcur'] = True
			while not activeNodes.empty():
				currentNode, timestep = activeNodes.get()
				for node in self.graph.successors(currentNode):
					if not self.graph.node[node]['isInfluencedSeed'] and not self.graph.node[node]['isInfluencedcur']:
						if isEnvironment:
							# dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
							dot = self.graph.edges[currentNode, node]['probabilities']
						else:
							# dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
							dot = self.graph.edges[currentNode, node]['estimates']
						if dot > possibleWorld[currentNode][node]:
							activeNodes.put([node, timestep+1])
							self.graph.node[node]['isInfluencedcur'] = True

		influencedNodes = []
		influencedNodesCur = []
		influencedNodesu = []
		influencedNodesuCur = []
		for n in self.graph.nodes:
			if self.graph.node[n]['isInfluencedSeed']:
				influencedNodes.append(n)
			if self.graph.node[n]['isInfluencedSeed'] or self.graph.node[n]['isInfluencedu']:
				influencedNodesu.append(n)
			if self.graph.node[n]['isInfluencedSeed'] or self.graph.node[n]['isInfluencedcur']:
				influencedNodesCur.append(n)
			if self.graph.node[n]['isInfluencedSeed'] or self.graph.node[n]['isInfluencedcur'] or self.graph.node[n]['isInfluencedu']:
				influencedNodesuCur.append(n)
		return len(influencedNodes), len(influencedNodesu), len(influencedNodesCur), len(influencedNodesuCur)

	def expectedSpread(self, seeds, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		for simulation in range(numberOfSimulations):
			expSpread += self.diffusion(seeds, isEnvironment)
		expSpread = expSpread/numberOfSimulations
		return expSpread

	def expectedSpreadCelf(self, seeds, u, cur_best, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		expSpreadu = 0.0
		expSpreadCur = 0.0
		expSpreaduCur = 0.0
		for simulation in range(numberOfSimulations):
			spreads = self.diffusionCelf(seeds, u, cur_best, isEnvironment)
			expSpread += spreads[0]
			expSpreadu += spreads[1]
			expSpreadCur += spreads[2]
			expSpreaduCur += spreads[3]
		expSpread /= numberOfSimulations
		expSpreadu /= numberOfSimulations
		expSpreadCur /= numberOfSimulations
		expSpreaduCur /= numberOfSimulations
		return expSpread, expSpreadu, expSpreadCur, expSpreaduCur

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

	def celf(self, budget, isEnvironment=True, numberOfSimulations=100):
		seeds = []
		q = []
		seedMap = {}
		cur_best = None
		cur_best_spread = -1
		last_seed = None
		for node in self.graph.nodes:
			u = {}
			if cur_best is None:
				u['mg1'] = self.expectedSpread([node], numberOfSimulations, isEnvironment)
				u['mg2'] = u['mg1']
			else:
				spreads = self.expectedSpreadCelf([], node, cur_best, numberOfSimulations, isEnvironment)
				u['mg1'] = spreads[1] - spreads[0]
				u['mg2'] = spreads[3] - spreads[0]
			u['prev_best'] = cur_best
			u['flag'] = 0
			u['node'] = node
			heapq.heappush(q, (-u['mg1'], u))
			q.append((-u['mg1'], u))
			if cur_best_spread < u['mg1']:
				cur_best_spread = u['mg1']
				cur_best = node
		while len(seeds) < budget:
			u = heapq.heappop(q)[1]
			if u['flag'] == len(seeds):
				seeds.append(u['node'])
				last_seed = u['node']
				continue
			elif u['prev_best'] == last_seed:
				u['mg1'] = u['mg2']
			else:
				spreads = self.expectedSpreadCelf(seeds, u['node'], cur_best, numberOfSimulations, isEnvironment)
				u['mg1'] = spreads[1] - spreads[0]
				u['mg2'] = spreads[3] - spreads[2]
				u['prev_best'] = cur_best
			u['flag'] = len(seeds)
			if cur_best_spread < u['mg1']:
				cur_best_spread = u['mg1']
				cur_best = u['node']
			heapq.heappush(q, (-u['mg1'], u))
		return seeds

class MAB(object):
	def __init__(self, tic, budget):
		self.budget = budget
		self.tic = tic

	def explore(self):
		seeds = np.random.choice(self.tic.graph.number_of_nodes(), self.budget, replace=False)
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

	def L2Error(self):
		l2 = 0.
		for edge in self.tic.graph.edges:
			l2 += (self.tic.graph.edges[edge]['estimates'] - self.tic.graph.edges[edge]['probabilities'])**2
		return l2


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
			print self.tic.graph.edges[key]['probabilities'], self.tic.graph.edges[key]['estimates']

	def learnerNode(self, iterations, epsilon, gamma=0.1, initialEpochs = 10):
		S_w = {}
		positive_sum = {}
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
							if t > initialEpochs:
								if denominator > 1:
									self.tic.graph.edges[edge]['estimates'] = self.tic.graph.edges[edge]['estimates'] + ((self.tic.graph.edges[edge]['estimates']/P_w[edge[1]]) - self.tic.graph.edges[edge]['estimates'])/(denominator)
								else:
									self.tic.graph.edges[edge]['estimates'] = ((self.tic.graph.edges[edge]['estimates']/P_w[edge[1]]) - self.tic.graph.edges[edge]['estimates'])/(denominator)
						positive_sum[edge] = numerator
						
						key = edge
			print self.L2Error(), self.tic.graph.edges[key]['estimates'], self.tic.graph.edges[key]['probabilities'], self.tic.graph.number_of_edges()

		
itemList = []
numItems = 5
numTopics = 3
budget = 5
# for i in range(numItems):
# 	itemDist = np.random.rand(numTopics)
# 	itemList.append(Item(i, itemDist/sum(itemDist)))
tic = IC(generateGraph(100, density=0.1))
mab = MAB(tic, budget)
print mab.learnerNode(100000, 1)