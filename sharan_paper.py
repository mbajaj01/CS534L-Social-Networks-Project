import os
import sys
import snap
import numpy as np
import Queue
from scipy.misc import logsumexp
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self, id, inEdges, outEdges):
        self.id = id
        self.inEdges = inEdges
        self.outEdges = outEdges
        
    def printNode(self):
        print "Id" + str(self.id)
        print "OutEdges:"
        for edge in self.outEdges:
            edge.printEdge()
        print "InEdges:"
        for edge in self.inEdges:
            edge.printEdge()

class Edge(object):
    def __init__(self, id, node1, node2, ip):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.influenceProbability = ip
        self.estimate = np.random.rand()
        self.tried = 0
        self.activated = 0

    def printEdge(self):
        print "Id" + str(self.id) + " | Going " + str(self.node1.id) + " to " + str(self.node2.id) + " | IP" + str(self.influenceProbability)
        
        

class Graph(object):
    def __init__(self, nodes, edges, edgeIndices):
        self.edgeIndices = edgeIndices
        self.nodes = nodes
        self.edges = edges
        
    def printEdges(self):
        for edge in self.edges:
            edge.printEdge()
            
    def printNodes(self):
        for node in self.nodes:
            node.printNode()

    def getInfluenceProbabilities(self):
        ip = [edge.influenceProbability for edge in self.edges]
        return ip
    
    def getEstimates(self):
        ip = [edge.estimate for edge in self.edges]
        return ip
    
    def getIds(self, obj):
        return [i.id for i in obj]

class Item(object):
    def __init__(self, id, topicDistribution):
        self.id = id
        self.topicDistribution = topicDistribution
        self.topicEstimates = np.zeros(len(self.topicDistribution))


class Node2(object):
    def __init__(self, id, inNodes, outNodes, inProbs, outProbs):
        self.id = id
        self.inNodes = inNodes
        self.outNodes = outNodes
        self.inEdgesProbs = inProbs
        self.outEdgeProbs = outProbs

class CascadeLog(object):
    def __init__(self, node1, node2, edge, time):
        self.node1 = node1
        self.node2 = node2
        self.edge = edge
        self.time = time


def generateGraph(nodes, density=0.5):
    edgeProbs = np.random.rand(nodes,nodes)
    edgeOccurence = edgeProbs < density
    edgeIndices = np.transpose(edgeOccurence.nonzero())

    # Initialize graph
    graph = Graph([],[],edgeIndices)
    for node in range(nodes):
        graph.nodes.append(Node(node, [], []))

    edgeId = 0
    for id,edge in enumerate(edgeIndices):
        if (edge[0] == edge[1]):# to prevent self loops
            #print str(edge[0] )+ ' ' + str(edge[1])
            continue
        node1 = edge[0]
        node2 = edge[1]
        ip = np.random.rand()
        edge = Edge(edgeId,graph.nodes[node1],graph.nodes[node2],ip)
        graph.edges.append(edge)
        graph.nodes[node1].outEdges.append(edge)
        graph.nodes[node2].inEdges.append(edge)
        edgeId += 1
    return graph


from sets import Set
class Diffusion(object):
    def __init__(self, graph, seeds, mode):
        self.influencedNodes = Set([])
        self.traversedEdges = Set([])
        self.seeds = seeds
        self.graph = graph
        self.mode = mode
        #self.possibleWorld = self.generatePossibleWorldBase()
        self.possibleWorldProbs = self.generatePossibleWorld()
        self.PossibleWorld = []
    
    # TODOs
    ## remove mode from this class
    ## make a separate thing for walk and greedyAlgo for finding seeds
    
    def performDiffusion(self):
        possibleWorld = self.generatePossibleWorld()
        for node in self.seeds:
            if self.mode == 1:
                self.traverseIP(node)
            else:
                self.traverseEs(node)
        
    def generatePossibleWorld(self):
        coinFlips = np.random.rand(len(self.graph.edges))
        return coinFlips

    def generatePossibleWorldBase(self):
        coinFlips = np.zeros(len(self.graph.edges)).tolist()
        return coinFlips
    
    def traverseIP(self, node):
        self.influencedNodes.add(node.id)
        outEdges = node.outEdges
        for edge in outEdges:
            assert (node == edge.node1)
            if self.possibleWorldProbs[edge.id] <= edge.influenceProbability:
                if edge.node2.id not in self.influencedNodes:
                    self.traverseIP(edge.node2)

    def traverseEs(self, node):
        self.influencedNodes.add(node.id)
        outEdges = node.outEdges
        for edge in outEdges:
            assert (node == edge.node1)
            if self.possibleWorldProbs[edge.id] <= edge.estimate:
                if edge.node2.id not in self.influencedNodes: # should I use edges instead?
                    self.traverseEs(edge.node2)
            
    def walkPossibleWorld(self):
        possibleWorld = self.generatePossibleWorld()
        for node in self.seeds:
            self.traversePossibleWorld(node)
        
    def traversePossibleWorld(self, node):
        self.influencedNodes.add(node.id)
        outEdges = node.outEdges
        for edge in outEdges:
            assert (node == edge.node1)
            edge.tried += 1
            if self.possibleWorldProbs[edge.id] <= edge.influenceProbability:
                edge.activated += 1
                if edge.node2.id not in self.influencedNodes:
                    self.traversePossibleWorld(edge.node2)
                    

class IC(object):
    def __init__(self, graph, budget, noSimulations, mode):
        self.noSimulations = noSimulations
        self.graph = graph
        self.budget = budget
        self.maxSpread = 0.0
        self.seedSet = Set([])
        self.mode = mode
        
    def updateEstimates(self):
        for edge in self.graph.edges:
            if edge.tried >0:
                edge.estimate = float(edge.activated)/float(edge.tried)
        
    def expectedSpread(self, seeds):
        expSpread = 0.0
        for simulation in range(self.noSimulations):
            diffObj = Diffusion(self.graph, seeds, self.mode)
            diffObj.performDiffusion()
            expSpread += len(diffObj.influencedNodes)
        expSpread = expSpread/self.noSimulations
        return expSpread
    
    def findSpreadSeeds(self, seeds):
        diffObj = Diffusion(self.graph, seeds, self.mode)
        diffObj.walkPossibleWorld()
        self.maxSpread = len(diffObj.influencedNodes)
        self.seedSet = seeds
        
    def printStats(self):
        print "SeedSet: " + str([node.id for node in icDiffuser.seedSet])
        print "Spread: " + str(self.maxSpread)

    def findBestSpread(self):
        while len(self.seedSet) < self.budget:
            maxSpread = 0.0
            for candidate in self.graph.nodes:
                newSS = self.seedSet.union([candidate])
                spread = self.expectedSpread(newSS)
                if spread > maxSpread:
                    maxSpread = spread
                    maxNode = candidate
            self.maxSpread = maxSpread
            self.seedSet.add(maxNode)

    def computeEstimateRMSE(self):
        error = 0
        for edge in self.graph.edges:
            error += pow(edge.influenceProbability - edge.estimate,2)
        return pow(error/len(self.graph.edges),0.5)
            
class MAB:
    def __init__(self, graph, budget, mc):
        self.graph = graph
        self.budget = budget
        self.mc = mc
        self.diffusions = []
        self.estimatedRMSE = []
        self.estimatedSpreads = []
    
    def epsilonGreedy(self, e,rounds,mc):
        spread = {}
        for r in range(rounds):
            print 'Round ' + str(r)
            icDiffuser  = IC(self.graph, self.budget, mc, 0)
            if np.random.rand(1) > e:
                icDiffuserTemp  = IC(self.graph, self.budget, mc, 0)
                icDiffuserTemp.findBestSpread()
                seeds = icDiffuserTemp.seedSet
                icDiffuser.findSpreadSeeds(seeds)
            else:
                seedsInd = np.random.choice(len(self.graph.nodes), self.budget, replace=False)
                seeds = [self.graph.nodes[i] for i in seedsInd]
                icDiffuser.findSpreadSeeds(seeds)
            icDiffuser.updateEstimates()
            self.diffusions.append(icDiffuser)
            self.estimatedRMSE.append(icDiffuser.computeEstimateRMSE())
            self.estimatedSpreads.append(icDiffuser.maxSpread)
            
    def plotRegret(self):
        mc = 10000
        icDiffuser  = IC(self.graph, self.budget, mc, 1)
        icDiffuser.findBestSpread()
        bestSpread = icDiffuser.maxSpread
        spreadDiff = [bestSpread-i for i in self.estimatedSpreads]
        plt.plot(spreadDiff)
        plt.ylabel("Regret")
        plt.xlabel("Round")
        
    def plotSpreads(self):
        plt.plot(self.estimatedSpreads)
        plt.ylabel("Spread")
        plt.xlabel("Round")
        
            
    def plotRMSE(self):
        plt.plot(self.estimatedRMSE)
        plt.ylabel("RMSE")
        plt.xlabel("Round")
        plt.show()

import time
start_time = time.time()

noNodes = 10
density = 0.5
graph = generateGraph(noNodes, density)
budget = 2
mc = 5
e = 0.5
rounds = 500
m = MAB(graph, budget, mc)
m.epsilonGreedy(e, rounds, mc)

print("Execution time -  %s seconds" % (time.time() - start_time))

#print m.estimateRMSE
m.plotRMSE()