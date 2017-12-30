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

def generateGraphNew(nodes, density=0.5):
	G = nx.DiGraph()
	G.add_nodes_from(range(nodes))
	edgeProbs = np.random.rand(nodes,nodes)
	edgeOccurence = edgeProbs < density
	edgeIndices = np.transpose(edgeOccurence.nonzero())
	for id,edge in enumerate(edgeIndices):
		node1 = edge[0]
		node2 = edge[1]
		if node1 == node2:
			continue
		probabilities = 0.
		G.add_edges_from([(node1, node2, {'probabilities': probabilities, 'estimates': np.random.random(), 'average': 0})])
	numNodes = G.number_of_nodes()
	nodesCovered = 0.
	probD = {}
	for node in G.nodes:
		fracCovered = float(nodesCovered/numNodes)
		if fracCovered < 0.95:
			l = 0.1
		else:
			print "Node : ",node
			l = 0.9
		for edge in G.edges(node):
			probD[edge] = np.random.uniform(l, l+0.1)
		nodesCovered += 1
	nx.set_edge_attributes(G, probD, 'probabilities')
	return G

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


	def expectedSpreadCelf(self, seeds, u, cur_best, numberOfSimulations, isEnvironment):
		expSpread = 0.0
		expSpreadu = 0.0
		expSpreadCur = 0.0
		expSpreaduCur = 0.0
		for simulation in range(numberOfSimulations):
			self.initAllItems()
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

#here

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
				print newSeed
				seeds.append(newSeed)
				seedMap[newSeed] = 1
			else:
				break
		return seeds, maximumSpread

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
						try:
							self.R.edges[node, currentNode]['count'] += 1
						except:
							self.R.add_edges_from([(node,currentNode,{'count':1})])
		return wr

	def RRGreedy(self, budget, isEnvironment=True,  l=1., epsilon=0.2):
		self.R = nx.Graph()
		KPT = self.KPTEstimation(budget, isEnvironment, l, epsilon)
		n = self.graph.number_of_nodes()
		if self.lam == None:
			self.lam = (8+2*epsilon)*n*(l*np.log(n) + np.log(comb(n,budget)) + np.log(2))*(epsilon**-2)
		theta = int(np.ceil(self.lam/KPT))
		print "theta: ",theta
		theta = 10000
		for r in range(theta):
			RRset = self.generateRRSet(isEnvironment)
		#print "printing graph: "
		#print len(list(self.R.nodes)),"list: "
		#for node in self.R.nodes:
			#print node, self.R.degree(node) 
		seeds = []
		for i in range(budget):
			maxdegree = -1
			maxnode = -1
			for node in self.R.nodes:
				degree = 0
				for edge in self.R.edges(node):
					degree +=self.R.edges[edge]['count']
				if degree > maxdegree:
					maxnode = node
					maxdegree = degree
			if maxnode == -1:
				seeds.append(np.random.choice(self.graph.nodes,1)[0])
			else:
				self.R.remove_node(maxnode)
				seeds.append(maxnode)
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

	def epsilonGreedy(self, epsilon, seeds=None):
		if epsilon > np.random.rand(1):
			seeds = self.explore()
		else:
			if seeds is None:
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

	def verify_spread(self):
		nx.set_edge_attributes(self.tic.graph, {e: 0.8 for e in graph.edges}, 'estimates')
		#seedsTrue = self.tic.RRGreedy(self.budget, isEnvironment=True)
		seedsBandit = self.tic.RRGreedy(self.budget, isEnvironment=False)
		self.tic.initAllItems()
		#expSpreadTrue = self.tic.expectedSpread(seedsTrue, numberOfSimulations=50, isEnvironment=True)
		self.tic.initAllItems()
		expSpreadBandit = self.tic.expectedSpread(seedsBandit, numberOfSimulations=50, isEnvironment=True)
		#nx.set_edge_attributes(self.tic.graph, {e: np.random.random() for e in graph.edges}, 'estimates')
		#self.tic.initAllItems()
		#expSpreadBandit2 = self.tic.expectedSpread(seedsBandit, numberOfSimulations=50, isEnvironment=True)
		print expSpreadBandit

	def findTrueSeedsAndSpread(self):
		expSpreadTrue = 0.
		for i in range(3):
			seedsTrue = self.tic.RRGreedy(self.budget, isEnvironment=True)
			#print seedsTrue
			expSpreadTrue += self.tic.expectedSpread(seedsTrue, numberOfSimulations=500, isEnvironment=True)
			#print expSpreadTrue
		return expSpreadTrue/3

	def findTrueSeedsAndSpread2(self):
		expSpreadTrue = 0.
		for i in range(3):
			seedsTrue = self.tic.RRGreedy(self.budget, isEnvironment=True)
			print seedsTrue
			expSpreadTrue = self.tic.expectedSpread(seedsTrue, numberOfSimulations=500, isEnvironment=True)
			print expSpreadTrue
			self.tic.initAllItems()
			seedsTrue = self.tic.celf(self.budget,isEnvironment=True)
			print "celf: ",seedsTrue
			self.tic.initAllItems()
			expSpreadTrue = self.tic.expectedSpread(seedsTrue, numberOfSimulations=500, isEnvironment=True)
			print expSpreadTrue
			self.tic.initAllItems()
			seedsTrue,expSpreadTrue = self.tic.findBestSeeds(self.budget,isEnvironment=True,numberOfSimulations=10)
			print "*",seedsTrue, expSpreadTrue
			self.tic.initAllItems()
			expSpreadTrue += self.tic.expectedSpread(seedsTrue, numberOfSimulations=500, isEnvironment=True)
		return expSpreadTrue/3


	def learnerNode(self, iterations, epsilon, initialIter = 0):
		S_w = {}
		positive_sum = {}
		#seedsTrue = self.tic.RRGreedy(self.budget, isEnvironment=True)
		# expSpreadTrue = self.tic.expectedSpread(seedsTrue, 10, True)
		regret = 0.
		seedsBandit = self.tic.RRGreedy(self.budget, isEnvironment=False)
		#expSpreadTrue = 0.
		expSpreadBandit = 0.
		for diffIter in range(1):
			#possibleWorld = self.tic.generatePossibleWorld()
			self.tic.initAllItems()
			expSpreadBandit = self.tic.expectedSpread(seedsBandit, numberOfSimulations=100, isEnvironment=True)
		expSpreadTrue = self.findTrueSeedsAndSpread()

		#expSpreadTrue /= 1
		#expSpreadBandit /= 1
		print "Spreads: ",expSpreadTrue, expSpreadBandit
		regret += ( expSpreadTrue - expSpreadBandit)
		print "L2 error: ",self.L2Error(),"Regret: ",regret
		for t in range(iterations):
			if t == 100:
				epsilon=0.3
			self.tic.initAllItems()
			np.random.seed()
			seeds = self.epsilonGreedy(epsilon, seedsBandit)
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
				seedsBandit = self.tic.RRGreedy(self.budget, isEnvironment=False)
				#expSpreadTrue = 0.
				expSpreadBandit = 0.
				for diffIter in range(1):
					#possibleWorld = self.tic.generatePossibleWorld()
					#self.tic.initAllItems()
					#expSpreadTrue, possibleWorld = self.tic.diffusion(seedsTrue, isEnvironment=True, returnPossibleWorld=True)
					self.tic.initAllItems()
					#expSpreadBandit = self.tic.diffusion(seedsBandit, isEnvironment=True, possibleWorld=possibleWorld)
					expSpreadBandit = self.tic.expectedSpread(seedsBandit,numberOfSimulations=10, isEnvironment=True)
				#expSpreadTrue /= 1
				#expSpreadBandit /= 1
				print "Spreads: ",expSpreadTrue, expSpreadBandit
				regret += ( expSpreadTrue - expSpreadBandit)
				print "L2 error: ",self.L2Error(),"Regret: ",regret/(t+2)
			#print "L2 error: ",self.L2Error()

		
itemList = []
numItems = 5
numTopics = 3
budget = 5
# for i in range(numItems):
# 	itemDist = np.random.rand(numTopics)
# 	itemList.append(Item(i, itemDist/sum(itemDist)))
#graph = cPickle.load(open('data/wikiGraph.p'))
#nx.set_edge_attributes(graph, {e: np.random.random() for e in graph.edges}, 'estimates')
#nx.set_edge_attributes(graph, {e: np.random.random() for e in graph.edges}, 'probabilities')
graph = generateGraphNew(500, 0.01)
print graph.number_of_edges(), graph.number_of_nodes()
tic = IC(graph)
# tic.RRGreedy(budget)
mab = MAB(tic, budget)
#mab.verify_spread()
#mab.findTrueSeedsAndSpread()
print mab.learnerNode(100000, 0)