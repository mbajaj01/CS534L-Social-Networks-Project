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
import cPickle as pickle


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

	
	def EM(self, iterations, emIter, epsilon):
		
		positive_sum = {}
		S_w = {}
		for t in range(iterations):
			self.tic.initAllItems()
			np.random.seed()
			seeds = self.epsilonGreedy(epsilon)
			
			newIter = True
		for em_t in range(emIter):

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
						if newIter:
							S_w[(parent, node)]['positive'] += 1
						isPostiveParent[(parent, node)] = True
					else:
						if newIter:
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
						if em_t == emIter - 1:
							positive_sum[edge] = numerator
					self.tic.graph.edges[edge]['estimates'] =  numerator/denominator
					key = edge
			#for edge in self.tic.graph.edges:
			#randKey = np.random.randint(len(self.tic.graph.edges))
			print key, self.tic.graph.edges[key]['estimates'], self.tic.graph.edges[key]['probabilities']
			newIter = False


	#def setAttributesOnGraph(graph):


	def EMCascade(self, iterations,cascades,probEstimates):
		#annotate cascade into the graph
		S_w = {}
		edge_credit={}
		for c in range(cascades):
			for node in fplus[c]:
				for parent in fplus[c][node]:
					if (parent,node) not in S_w:
						S_w[(parent, node)] = {'positive':0, 'negative':0}
					S_w[(parent, node)]['positive'] += 1

			for node in fminus[c]:
				for parent in fminus[c][node]:
					if (parent,node) not in S_w:
						S_w[(parent, node)] = {'positive':0, 'negative':0}
					S_w[(parent, node)]['negative'] += 1


		for t in range(iterations):
			for c in range(cascades):
				

				
				cascade_prob = {}
				#for node in self.tic.graph.nodes:
				for node in fplus[c]:
					prob = 1.
					for parent in fplus[c][node]:
						cascade_prob[(parent,node)] = probEstimates[(parent, node)]
						prob *= 1. - probEstimates[parent, node]
						prob = 1. - prob
						if prob == 0.:
							prob = 1.

					for edge, credit in cascade_credit.items():
						if edge not in edge_credit:
							edge_credit[edge]=0
						edge_credit[edge]+=credit/prob

							#what about negative prob?
					#if self.tic.graph.node[node]['isInfluenced'] and self.tic.graph.node[node]['influenceTimestep'] > 0:
					#assuming up to be always true
			for edge,credit in edge_credit.items():
				denominator = S_w[edge]['positive'] + S_w[edge]['negative']
				probEstimates[edge] = edge_credit/denominator
		for edge, prob in probEstimates.items():
			print edge,prob
					

	def learnerNode(self, iterations, epsilon):
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
						#what about negative prob?
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
						positive_sum[edge] = numerator
					self.tic.graph.edges[edge]['estimates'] =  numerator/denominator
					key = edge
			print key, self.tic.graph.edges[key]['estimates'], self.tic.graph.edges[key]['probabilities'] 




def logLikelihoodPrint(fplus,fminus,probEstimates):

	logTerm1=0.
	logTerm2=0.
	logProd=0.
	for c in range(cascades):
		for node in fplus[c]:
			prob_product=1.
			for parent in fplus[c][node].keys():
				prob = 1. - probEstimates[(parent, node)]
				prob_product *= prob
			if prob_product == 1.:
				print "likelihood error"
				return
			prob_product = 1. - prob_product
			logTerm1 += np.log(prob_product)

	for c in range(cascades):
		for node in fminus[c]:
			prob_product=1.
			logProd = 0.
			for parent in fminus[c][node].keys():
				prob = 1. - probEstimates[(parent, node)]
				logProd += np.log(prob)
			logTerm2 += logProd

	logHood = logTerm2 + logTerm1
	print "logLikelihood: ",logHood




def EMCascade(iterations,cascades,fplus,fminus,probEstimates):
		#annotate cascade into the graph
		S_w = {}
		
		for c in range(cascades):
			for node in fplus[c]:
				for parent in fplus[c][node].keys():
					if (parent,node) not in S_w:
						S_w[(parent, node)] = {'positive':0, 'negative':0}
					S_w[(parent, node)]['positive'] += 1

			for node in fminus[c]:
				for parent in fminus[c][node].keys():
					if (parent,node) not in S_w:
						S_w[(parent, node)] = {'positive':0, 'negative':0}
					S_w[(parent, node)]['negative'] += 1


		for t in range(iterations):
			edge_credit={}
			for c in range(cascades):
				

				
				
				#for node in self.tic.graph.nodes:
				for node in fplus[c]:
					cascade_prob = {}
					prob = 1.
					for parent in fplus[c][node].keys():
						cascade_prob[(parent,node)] = probEstimates[(parent,node)]
						prob *= (1. - probEstimates[(parent, node)])
					prob = 1. - prob
						#if prob == 0.:
						#		prob = 1.

					for edge, credit in cascade_prob.items():
						if edge not in edge_credit:
							edge_credit[edge]=0
						credit_norm = credit/prob
						#if credit_norm > 1:
						#	print "Error",credit_norm
						edge_credit[edge]+=credit_norm

							#what about negative prob?
					#if self.tic.graph.node[node]['isInfluenced'] and self.tic.graph.node[node]['influenceTimestep'] > 0:
					#assuming up to be always true
			for edge,credit in edge_credit.items():
				denominator = S_w[edge]['positive'] + S_w[edge]['negative']
				if denominator < credit:
					print edge
					print ("Error")
					print "num/den",credit,denominator
					return
				
				probEstimates[edge] = credit/denominator
				#print edge, (probEstimates[edge])
			logLikelihoodPrint(fplus,fminus,probEstimates)


fplus={}
r=12
for i in range(r):
	l=str(200*i)
	h=str(200*i+199)
	file = 'Fdata_inf/Fplus_'+l+'_'+h
	d=pickle.load( open(file,"rb"))
	for i in d.keys():
		fplus[i]=d[i]
#file = 'Fdata_inf/Fplus_end'
#d=pickle.load( open(file,"rb"))
#for i in d.keys():
#	fplus[i]=d[i]

fminus={}
for i in range(r):
	l=str(200*i)
	h=str(200*i+199)
	file = 'Fdata_inf/Fminus_'+l+'_'+h
	d=pickle.load( open(file,"rb"))
	for i in d.keys():
		fminus[i]=d[i]


#file = 'Fdata_inf/Fminus_end'
#d=pickle.load( open(file,"rb"))
#for i in d.keys():
#	fplus[i]=d[i]
probEstimates={}
cascades=r*200
for c in range(cascades):
	for node in fplus[c]:
		for parent in fplus[c][node].keys():
			probEstimates[(parent,node)]=np.random.rand(1)

for c in range(cascades):
	for node in fminus[c]:
		for parent in fminus[c][node].keys():
			probEstimates[(parent,node)]=np.random.rand(1)

print "data loaded"

probEstimates=EMCascade(50,cascades,fplus,fminus,probEstimates)

#itemList = []
#numItems = 5
#numTopics = 3
#budget = 5
# for i in range(numItems):
# 	itemDist = np.random.rand(numTopics)
# 	itemList.append(Item(i, itemDist/sum(itemDist)))
#G = nx.read_gpickle('data/graph.p')


#tic = IC(generateGraph(100, density=0.01))
#mab = MAB(tic, budget)
#print mab.learnerNode(10000, 1)
#print mab.EM(1000, 50, 1)