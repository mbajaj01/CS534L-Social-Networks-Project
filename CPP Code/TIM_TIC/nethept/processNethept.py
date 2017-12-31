import sys
import os
import numpy as np

numTopics = 10
outputFile = open('graph_tic_'+str(numTopics)+'.inf','w')
with open('graph_ic.inf') as inputFile:
	for line in inputFile:
		line = line.split()
		prob = np.random.uniform(low=0.0, high=float(line[2]), size=(numTopics))
		out = line[0] + " " + line[1] + " " + (" ").join([str(np.around(p,8)) for p in prob])
		outputFile.write(out + "\n")
		print out
outputFile.close()