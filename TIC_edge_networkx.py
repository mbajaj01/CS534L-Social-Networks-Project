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

def show_tree(tree, total_width=36, fill=' '):
    """Pretty-print a tree."""
    output = StringIO()
    last_row = -1
    for i, n in enumerate(tree):
        if i:
            row = int(math.floor(math.log(i+1, 2)))
        else:
            row = 0
        if row != last_row:
            output.write('\n')
        columns = 2**row
        col_width = int(math.floor((total_width * 1.0) / columns))
        output.write(str(n).center(col_width, fill))
        last_row = row
    print output.getvalue()
    print '-' * total_width
    print
    return

def generateGraph(nodes, topics, density=0.5):
	G = nx.DiGraph()
	G.add_nodes_from(range(nodes))
	edgeProbs = np.random.rand(nodes,nodes)
	edgeOccurence = edgeProbs < density
	edgeIndices = np.transpose(edgeOccurence.nonzero())
	for id,edge in enumerate(edgeIndices):
		node1 = edge[0]
		node2 = edge[1]
		probabilities = np.random.rand(topics)
		G.add_edges_from([(node1, node2, {'probabilities': probabilities, 'estimates': np.random.rand(topics)})])
	return G

class TIC(object):
	def __init__(self, graph, numTopics, items):
		self.graph = graph
		self.numTopics = numTopics
		self.numItems = len(items)
		self.items = items

	def generatePossibleWorld(self):
		coinFlips = np.random.rand(self.graph.number_of_nodes(), self.graph.number_of_nodes())
		return coinFlips

	def diffusion(self, seeds, item, isEnvironment):
		possibleWorld = self.generatePossibleWorld()
		activeNodes = Queue.Queue()
		nx.set_node_attributes(self.graph, False, 'isInfluenced')
		nx.set_node_attributes(self.graph, None, 'influencedBy')
		nx.set_node_attributes(self.graph, None, 'influenced')
		nx.set_node_attributes(self.graph, None, 'influencedNeighbours')
		nx.set_edge_attributes(self.graph, False, 'isActive')
		for seed in seeds:
			activeNodes.put([seed, 0])
			self.graph.node[seed]['isInfluenced'] = True
			self.graph.node[seed]['influencedBy'] = -1
			self.graph.node[seed]['influenced'] = []
			self.graph.node[seed]['influencedNeighbours'] = {}
		while not activeNodes.empty():
			currentNode, timestep = activeNodes.get()
			for node in self.graph.successors(currentNode):
				self.graph.node[seed]['influencedNeighbours'][currentNode] = True
				if not self.graph.node[node]['isInfluenced']:
					if isEnvironment:
						dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
					else:
						dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
					if dot > possibleWorld[currentNode][node]:
						activeNodes.put([node, timestep+1])
						self.graph.node[node]['isInfluenced'] = True
						self.graph.node[node]['influencedBy'] = currentNode
						self.graph.node[node]['influenced'] = []
						self.graph.node[currentNode]['influenced'].append(node)
						self.graph.edges[currentNode, node]['isActive'] = True

		influencedNodes = [n for n in self.graph.nodes if self.graph.node[n]['isInfluenced'] == True]
		return len(influencedNodes)

	def diffusionCelf(self, seeds, u, cur_best, item, isEnvironment):
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
						dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
					else:
						dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
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
							dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
						else:
							dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
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
							dot = np.dot(self.graph.edges[currentNode, node]['probabilities'], item.topicDistribution)
						else:
							dot = np.dot(self.graph.edges[currentNode, node]['estimates'], item.topicEstimates)	
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

	def expectedSpread(self, seeds, item, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		for simulation in range(numberOfSimulations):
			expSpread += self.diffusion(seeds, item, isEnvironment)
		expSpread = expSpread/numberOfSimulations
		return expSpread

	def expectedSpreadCelf(self, seeds, u, cur_best, item, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		expSpreadu = 0.0
		expSpreadCur = 0.0
		expSpreaduCur = 0.0
		for simulation in range(numberOfSimulations):
			spreads = self.diffusionCelf(seeds, u, cur_best, item, isEnvironment)
			expSpread += spreads[0]
			expSpreadu += spreads[1]
			expSpreadCur += spreads[2]
			expSpreaduCur += spreads[3]
		expSpread /= numberOfSimulations
		expSpreadu /= numberOfSimulations
		expSpreadCur /= numberOfSimulations
		expSpreaduCur /= numberOfSimulations
		return expSpread, expSpreadu, expSpreadCur, expSpreaduCur

	def findBestSeeds(self, item, budget, isEnvironment=True, numberOfSimulations=100):
		seeds = []
		seedMap = {}
		while len(seeds) < budget:
			maximumSpread = 0
			newSeed = -1
			for candidate in range(self.graph.number_of_nodes()):
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

	def celf(self, item, budget, isEnvironment=True, numberOfSimulations=100):
		seeds = []
		q = []
		seedMap = {}
		cur_best = None
		cur_best_spread = -1
		last_seed = None
		for node in self.graph.nodes:
			u = {}
			if cur_best is None:
				u['mg1'] = self.expectedSpread([node], item, numberOfSimulations, isEnvironment)
				u['mg2'] = u['mg1']
			else:
				spreads = self.expectedSpreadCelf([], node, cur_best, item, numberOfSimulations, isEnvironment)
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
				spreads = self.expectedSpreadCelf(seeds, u['node'], cur_best, item, numberOfSimulations, isEnvironment)
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

	def explore(self, item):
		seeds = np.random.choice(self.tic.graph.number_of_nodes(), self.budget, replace=False)
		return seeds.tolist()

	def exploit(self, item):
		seeds = self.tic.celf(item, self.budget, isEnvironment=False)
		return seeds

	def epsilonGreedy(self, epsilon, item):
		if epsilon > np.random.rand(1):
			seeds = self.explore(item)
		else:
			seeds = self.exploit(item)
		spread = self.tic.diffusion(seeds=seeds, item=item, isEnvironment=True)
		return seeds

	def learner(self, iterations, epsilon):
		pi = np.random.rand(self.tic.numTopics)
		pi = np.log(pi/sum(pi))
		nx.set_edge_attributes(self.tic.graph, {e:np.zeros(self.tic.numItems) for e in self.tic.graph.edges}, 'totalActivations')
		for t in range(iterations):
			for i in range(self.tic.numItems):
				seeds = self.epsilonGreedy(epsilon, self.tic.items[i])
				cascadeLogProbPositive = np.zeros(self.tic.numTopics)
				cascadeLogProbNegative = np.zeros(self.tic.numTopics)
				for node1,node2 in self.tic.graph.edges:
					mask = self.tic.graph.edges[node1, node2]['estimates'] == 0.
					negmask = self.tic.graph.edges[node1, node2]['estimates'] == 1.
					if self.tic.graph.edges[node1, node2]['isActive']:
						cascadeLogProbPositive += np.log(self.tic.graph.edges[node1, node2]['estimates'])
						self.tic.graph.edges[node1, node2]['totalActivations'][i] += 1
					else:
						cascadeLogProbPositive += np.log(1. - self.tic.graph.edges[node1, node2]['estimates'])

				self.tic.items[i].topicEstimates = (pi + cascadeLogProbPositive + cascadeLogProbNegative)
				norm = logsumexp(self.tic.items[i].topicEstimates)

				if norm != -np.inf:
					self.tic.items[i].topicEstimates = np.exp(self.tic.items[i].topicEstimates - norm)
				else:
					self.tic.items[i].topicEstimates = np.zeros(self.tic.numTopics)
			pi = np.sum([self.tic.items[i].topicEstimates for i in range(self.tic.numItems)],0)
			normalizer = pi[:]
			pi = pi/self.tic.numItems
			mask = pi == 0.
			normalizer[mask] += 1
			for node1, node2 in self.tic.graph.edges:
				if self.tic.graph.edges[node1, node2]['isActive']:	
					self.tic.graph.edges[node1, node2]['estimates'] = np.sum([(self.tic.items[i].topicEstimates * self.tic.graph.edges[node1, node2]['totalActivations'][i])/(t+1) for i in range(self.tic.numItems)],0)/normalizer
			
		print self.tic.items[0].topicEstimates, self.tic.items[0].topicDistribution

itemList = []
numItems = 2
numTopics = 3
for i in range(numItems):
	itemDist = np.random.rand(numTopics)
	itemList.append(Item(i, itemDist/sum(itemDist)))
tic = TIC(generateGraph(100, numTopics, density=0.01), numTopics, itemList)
mab = MAB(tic, numTopics)
print mab.learner(10000, 1)