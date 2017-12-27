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
import cPickle
import dateutil.parser
import datetime

np.seterr(all='raise')

def loadGraph(path, numTopics):
	graph = cPickle.load(open(path))
	nx.set_edge_attributes(graph, {e: np.random.rand(numTopics) for e in graph.edges}, 'probabilities')
	return graph

def loadCascade(path):
	with open(path) as inputFile:
		header = inputFile.readline()
		cascade = {}
		for line in inputFile:
			line = line.strip().split(",")
			timestamp = datetime.datetime.fromtimestamp(int(line[1]))
			nodeID = int(line[2])
			itemID = int(line[3])
			date = dateutil.parser.parse(line[4])
			if itemID not in cascade:
				cascade[itemID] = []
			cascade[itemID].append((timestamp, nodeID, itemID))
	cPickle.dump(cascade, open('data/cascade.p', 'wb'),-1)

def sortCascade(path):
	cascade = cPickle.load(open(path))
	cascadeSorted = {}
	for item in cascade.keys():
		cascadeSorted[item] = sorted(cascade[item], key=lambda tup: tup[0])
	cPickle.dump(cascadeSorted, open('data/cascadeSorted.p', 'wb'), -1)


class TIC(object):
	def __init__(self, graph, numTopics):
		self.graph = graph
		self.numTopics = numTopics

	def generatePossibleWorld(self):
		coinFlips = np.random.rand(self.graph.number_of_nodes(), self.graph.number_of_nodes())
		return coinFlips

	def generateCascade(self, path, delta):
		cascade = cPickle.load(open(path))
		nodeset = {n:1 for n in self.graph.nodes}
		Fplus = {}
		Fminus = {}
		Splus = {}
		Sminus = {}
		self.numItems = len(cascade.keys())
		for i in range(self.numItems):
			print i
			Fplus[i] = {}
			Fminus[i] = {}
			nodeinfo = {}
			for log in cascade[i+1]:
				date = log[0]
				n = log[1]
				nodeinfo[n] = date
			for node in self.graph.nodes:
				for parent in self.graph.predecessors(node):
					if parent in nodeinfo:
						if node in nodeinfo:
							if nodeinfo[parent] <= nodeinfo[node]:
								if nodeinfo[node] <= (nodeinfo[parent] + datetime.timedelta(hours=delta)):
									if node not in Fplus[i]:
										Fplus[i][node] = {}
									if (parent, node) not in Splus:
										Splus[(parent, node)] = {}
									if i in Splus[(parent, node)]:
										print "Error"
										sys.exit()
									Splus[(parent, node)][i] = 1
									Fplus[i][node][parent] = nodeinfo[node] - nodeinfo[parent]
								else:
									if node not in Fminus[i]:
										Fminus[i][node] = {}
									print "error"
									sys.exit()
									if (parent, node) not in Sminus:
										Sminus[(parent, node)] = {}
									if i in Sminus[(parent, node)]:
										print "Error"
										sys.exit()
									Sminus[(parent, node)][i] = 1
									Fminus[i][node][parent] = nodeinfo[node] - nodeinfo[parent]
						else:
							if node not in Fminus[i]:
								Fminus[i][node] = {}
							if (parent, node) not in Sminus:
								Sminus[(parent, node)] = {}
							if i in Sminus[(parent, node)]:
								print "Error"
								sys.exit()
							Sminus[(parent, node)][i] = 1
							Fminus[i][node][parent] = np.inf
			if i%200 == 199:
				name = "_"+str(i-200+1) + "_" + str(i)
				cPickle.dump(Fplus, open('data/Fdata_inf/Fplus'+name,'wb'), -1)
				cPickle.dump(Fminus, open('data/Fdata_inf/Fminus'+name,'wb'), -1)
				del Fplus
				del Fminus
				Fplus = {}
				Fminus = {}

		cPickle.dump(Fplus, open('data/Fdata_inf/Fplus_end','wb'), -1)
		cPickle.dump(Fminus, open('data/Fdata_inf/Fminus_end','wb'), -1)
		cPickle.dump(Splus, open('data/Fdata_inf/Splus.p','wb'), -1)
		cPickle.dump(Sminus, open('data/Fdata_inf/Sminus.p','wb'), -1)

	def EM(self, directory, EMIter = 40):
		pi = np.random.rand(self.numTopics)
		pi = np.log(pi/sum(pi))
		name = "_0_199"
		Fplus = cPickle.load(open(directory+"Fplus"+name))
		Fminus = cPickle.load(open(directory+"Fminus"+name))
		Splus = cPickle.load(open(directory+"Splus.p"))
		Sminus = cPickle.load(open(directory+"Sminus.p"))
		nodeset = {}
		R = {}
		self.items = {}
		self.numItems = len(Fplus.keys())
		for item in Fplus.keys():
			nodeset[item] = {}
			itemDist = np.random.rand(self.numTopics)
			R[item] = {}
			self.items[item] = Item(item, itemDist/sum(itemDist))
			for node in Fplus[item].keys():
				nodeset[item][node] = 1
			for node in Fminus[item].keys():
				nodeset[item][node] = 1

		for iteration in range(EMIter):
			print iteration
			logLikelihood = np.zeros(shape=(self.numItems, self.numTopics))
			itemSum = np.zeros(self.numTopics)
			for item in Fplus.keys():
				cascadeLogProbPositive = np.zeros(self.numTopics)
				cascadeLogProbNegative = np.zeros(self.numTopics)
				for node in nodeset[item]:
					cascadeProbPositive = np.ones(self.numTopics)
					influenced = 0
					if node in Fplus[item]:
						influenced = 1
						for positivenode in Fplus[item][node].keys():
							cascadeProbPositive *= 1. - self.graph.edges[positivenode, node]['probabilities']
							if iteration > 0:
								try:
									logLikelihood[item,:] += (R[item][(positivenode, node)]*np.log(self.graph.edges[positivenode, node]['probabilities'])) + ((1 - R[item][(positivenode, node)])*(np.log(1. - self.graph.edges[positivenode, node]['probabilities'])))
								except:
									edgeProb = np.copy(self.graph.edges[positivenode, node]['probabilities'])
									edgeProb[edgeProb < 10**-50] = 10**-50
									try:
										logLikelihood[item,:] += (R[item][(positivenode, node)]*np.log(edgeProb)) + ((1 - R[item][(positivenode, node)])*(np.log(1. - edgeProb)))
									except:
										negProb = np.copy(edgeProb)
										posProb = np.copy(1.- edgeProb)
										posProb[posProb <= 0] = 1.
										negProb[negProb <= 0] = 1.
										logLikelihood[item,:] += (R[item][(positivenode, node)]*np.log(posProb)) + ((1 - R[item][(positivenode, node)])*(np.log(negProb)))
					
					if node in Fminus[item]:
						for negativeNode in Fminus[item][node].keys():
							negative = 1. - self.graph.edges[negativeNode, node]['probabilities']
							try:
								cascadeLogProbNegative += np.log(negative)
							except:
								negative[negative <= 0.] = 1.
								cascadeLogProbNegative += np.log(negative)
							logLikelihood[item, :] += np.log(negative)
					if influenced != 0:
						cascadeProbPositive = 1. - cascadeProbPositive
						try:
							cascadeLogProbPositive += np.log(cascadeProbPositive)
						except:
							cascadeProbPositive[cascadeProbPositive <= 0.] = 1.
							cascadeLogProbPositive += np.log(cascadeProbPositive)

						for positivenode in Fplus[item][node].keys():
							try:
								R[item][(positivenode, node)] = self.graph.edges[positivenode, node]['probabilities']/(cascadeProbPositive)
							except:
								cascadeProbPositive[cascadeProbPositive < 10**-50] = 10**-50
							R[item][(positivenode, node)][R[item][(positivenode, node)] > 1.] = 1.
							if sum(R[item][(positivenode, node)] < 0.) > 0:
								print "Error", R[item][(positivenode, node)]
				logLikelihood[item, :] += np.sum(pi)
				logLikelihood[item, :] *= self.items[item].topicDistribution 
				self.items[item].topicDistribution =  pi + cascadeLogProbPositive + cascadeLogProbNegative
				try:
					norm = logsumexp(self.items[item].topicDistribution)
				except:
					self.items[item].topicDistribution[self.items[item].topicDistribution < -500] = -500
					norm = logsumexp(self.items[item].topicDistribution)
				self.items[item].topicDistribution = np.exp(self.items[item].topicDistribution - norm)
				itemSum += np.copy(self.items[item].topicDistribution)
			if iteration > 0:
				print "Likelihood:", np.sum(logLikelihood)
			pi = np.log(itemSum/self.numItems)
			for node1, node2 in Splus.keys():
				numerator = np.zeros(self.numTopics)
				denominator = np.zeros(self.numTopics)
				update = 0
				for item in Splus[(node1, node2)].keys():
					if item in Fplus:
						try:
							numerator += self.items[item].topicDistribution * R[item][(node1, node2)]
							denominator += self.items[item].topicDistribution
							if numerator > denominator:
								print "Error New"
								print numerator, denominator,  R[item][(node1, node2)]
								sys.exit()
						except:
							R[item][(node1, node2)][R[item][(node1, node2)] < 10**-50] = 10**-50.
							try:
								numerator += self.items[item].topicDistribution * R[item][(node1, node2)]
								denominator += self.items[item].topicDistribution
							except:
								dist = np.copy(self.items[item].topicDistribution)
								dist[dist < 10**-50] = 10**-50.
								numerator += dist * R[item][(node1, node2)]
								denominator += dist
						update = 1

				if update == 1:
					if (node1, node2) in Sminus:
						for item in Sminus[(node1, node2)].keys():
							if item in Fminus:
								denominator += self.items[item].topicDistribution

					try:
						self.graph.edges[node1, node2]['probabilities'] = numerator/denominator
					except:
						numerator[numerator < 10**-50] = 10**-50.
						try:
							self.graph.edges[node1, node2]['probabilities'] = numerator/denominator
						except:
							denominator[denominator < 10**-50] = 10**-50.
							self.graph.edges[node1, node2]['probabilities'] = numerator/denominator	

					probMask = (self.graph.edges[node1, node2]['probabilities'] > 1.)
					self.graph.edges[node1, node2]['probabilities'][probMask] = 1.
					if sum(self.graph.edges[node1, node2]['probabilities'] < 0.) > 0:
						print "Error: ", numerator, denominator, self.graph.edges[node1, node2]['probabilities']
						sys.exit()


numTopics = 3
graph = loadGraph('data/graph.p', numTopics)
tic = TIC(graph, numTopics)
# tic.generateCascade('data/cascadeSorted.p',10000000)
tic.EM('data/Fdata_inf/')
#loadCascade('data/digg_votes_pruned.csv')
