import matplotlib.pyplot as plt

########## NETHEPT ##############
baseline_highdeg='nethept_highdegree_50seeds_1000iter'
baseline_random='nethept_random_50seeds_1000iter'

#TIC node
# exploit_file = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_10topics'
# ep_greedy = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_10topics'
# explore_file = 'nethept_tic_node_Explore_50seeds_1000iter_10items_10topics'
# name = 'tic_nethept_node_10_10'

# exploit_file = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_5topics'
# ep_greedy = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_5topics'
# explore_file = 'nethept_tic_node_Explore_50seeds_1000iter_10items_5topics'
# name = 'tic_nethept_node_10_5'

# exploit_file = 'nethept_tic_node_Exploit_50seeds_1000iter_5items_10topics'
# ep_greedy = 'nethept_tic_node_EGreedy_50seeds_1000iter_5items_10topics'
# explore_file = 'nethept_tic_node_Explore_50seeds_1000iter_5items_10topics'
# name = 'tic_nethept_node_5_10'

#TIC edge
# exploit_file = 'nethept_tic_edge_Exploit_50seeds_1000iter_10items_10topics'
# ep_greedy = 'nethept_tic_edge_EGreedy_50seeds_1000iter_10items_10topics'
# explore_file = 'nethept_tic_edge_Explore_50seeds_1000iter_10items_10topics'
# name = 'tic_nethept_edge_10_10'

# exploit_file = 'nethept_tic_edge_Exploit_50seeds_1000iter_10items_5topics'
# ep_greedy = 'nethept_tic_edge_EGreedy_50seeds_1000iter_10items_5topics'
# explore_file = 'nethept_tic_edge_Explore_50seeds_1000iter_10items_5topics'
# name = 'tic_nethept_edge_10_5'

# exploit_file = 'nethept_tic_edge_Exploit_50seeds_1000iter_5items_10topics'
# ep_greedy = 'nethept_tic_edge_EGreedy_50seeds_1000iter_5items_10topics'
# explore_file = 'nethept_tic_edge_Explore_50seeds_1000iter_5items_10topics'
# name = 'tic_nethept_edge_5_10'


########## SYNTH ##############
# baseline_highdeg='synth_highDegree_50seeds_1000iter'
# baseline_random='synth_random_50seeds_1000iter'


#TIC edge
# exploit_file = 'synth_tic_edge_Exploit_50seeds_1000iter_10items_10topics'
# ep_greedy = 'synth_tic_edge_EGreedy_50seeds_1000iter_10items_10topics'
# explore_file = 'synth_tic_edge_Explore_50seeds_1000iter_10items_10topics'
# name = 'tic_synth_edge_10_10'

#TIC node
# exploit_file = 'synth_tic_node_Exploit_50seeds_1000iter_10items_10topics'
# ep_greedy = 'synth_tic_node_EGreedy_50seeds_1000iter_10items_10topics'
# explore_file = 'synth_tic_node_Explore_50seeds_1000iter_10items_10topics'
# name = 'tic_synth_node_10_10'

################## PLOTING CODE #######################
strdata1 = open(baseline_highdeg, 'rb').read().split('\n')
strdata2 = open(baseline_random, 'rb').read().split('\n')
size = len(strdata1)


regret1 = []
regret2 = []
regret1_marker = []
regret2_marker = []
idregret = 1
iterations_markers_baseline = []
for i,datum in enumerate(strdata1):
	if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
		try:
			regret1_marker.append(float(strdata1[i].split(',')[idregret]))
			regret2_marker.append(float(strdata2[i].split(',')[idregret]))
			iterations_markers_baseline.append(i+1)
		except:
			print 'could not parse this -> ' + datum + strdata1[i] + strdata2[i]
	try:
		regret1.append(float(strdata1[i].split(',')[idregret]))
		regret2.append(float(strdata2[i].split(',')[idregret]))
	except:
		print 'could not parse this -> ' + strdata1[i] + strdata2[i]

# print regret1_marker
# print regret1
#exit()

highdegregret = regret1
randomregret = regret2
highdegregret_markers = regret1_marker
randomregret_markers = regret2_marker

print len(highdegregret)
print len (randomregret)
print len(highdegregret_markers)
print len(randomregret_markers)

### MAIN CODE ####

strdata1 = open(exploit_file, 'rb').read().split('\n')
strdata2 = open(ep_greedy, 'rb').read().split('\n')
strdata3 = open(explore_file, 'rb').read().split('\n')
size = len(strdata1)

l11 = []
l12 = []
l13 = []
l11_markers = []
l12_markers = []
l13_markers = []
idl1 = 0
dotl11 = []
dotl12 = []
dotl13 = []
dotl11_markers = []
dotl12_markers = []
dotl13_markers = []
dotidl1 = 1
regret1 = [] 
regret2 = []
regret3 = []
regret1_markers = [] 
regret2_markers = []
regret3_markers = []
idregret = 2
iterations = []
iterations_markers = []
count = 0
count_markers = 0
sep = ','

print len(highdegregret)

for i,datum in enumerate(strdata1):
	if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
		if count_markers < len(highdegregret_markers):
			try:
				l11_markers.append(float(strdata1[i].split(sep)[idl1]))
				l12_markers.append(float(strdata2[i].split(sep)[idl1]))
				l13_markers.append(float(strdata3[i].split(sep)[idl1]))
				dotl11_markers.append(float(strdata1[i].split(sep)[dotidl1]))
				dotl12_markers.append(float(strdata2[i].split(sep)[dotidl1]))
				dotl13_markers.append(float(strdata3[i].split(sep)[dotidl1]))
				regret1_markers.append(float(strdata1[i].split(sep)[idregret])/(i+1))
				regret2_markers.append(float(strdata2[i].split(sep)[idregret])/(i+1))
				regret3_markers.append(float(strdata3[i].split(sep)[idregret])/(i+1))
				iterations_markers.append(i+1)
				count_markers = count_markers + 1
			except:
				print 'could not parse this -> ' + strdata1[i] + strdata2[i] + strdata3[i] 
	
	if count < len(highdegregret):
		try:
			l11.append(float(strdata1[i].split(sep)[idl1]))
			l12.append(float(strdata2[i].split(sep)[idl1]))
			l13.append(float(strdata3[i].split(sep)[idl1]))
			dotl11.append(float(strdata1[i].split(sep)[dotidl1]))
			dotl12.append(float(strdata2[i].split(sep)[dotidl1]))
			dotl13.append(float(strdata3[i].split(sep)[dotidl1]))
			regret1.append(float(strdata1[i].split(sep)[idregret])/(i+1))
			regret2.append(float(strdata2[i].split(sep)[idregret])/(i+1))
			regret3.append(float(strdata3[i].split(sep)[idregret])/(i+1))
			iterations.append(i+1)
			count = count + 1
		except:
			print 'could not parse this -> ' + strdata1[i] + strdata2[i] + strdata3[i] 



print len(l11)
print len(l12)
print len(l13)
print len(regret1)
print len(regret2)
print len(regret3)
print len(l11_markers)
print len(l12_markers)
print len(l13_markers)
print len(regret1_markers)
print len(regret2_markers)
print len(regret3_markers)
print len(iterations)

# randomregret = randomregret[0:len(iterations)]
# highdegregret = highdegregret[0:len(iterations)]
# iterations = iterations[0:len(randomregret)]
# l11 = l11[0:len(randomregret)]
# l12 = l12[0:len(randomregret)]
# l13 = l13[0:len(randomregret)]
# regret1 = regret1[0:len(randomregret)]
# regret2 = regret2[0:len(randomregret)]
# regret3 = regret3[0:len(randomregret)]

print len (iterations)
print len (randomregret)

#L1

plt.plot(iterations,l11,'b')
plt.plot(iterations,l12,color='DarkOrange')
plt.plot(iterations,l13,'g')
plt.grid(True)
plt.legend(['Pure exploitation','Epsilon - 0.5','Pure exploration'])
plt.xlabel('Rounds')
plt.ylabel('Relative error')
plt.title('Relative error vs rounds')

# plot markers
plt.plot(iterations_markers,l11_markers,'bo')
plt.plot(iterations_markers,l12_markers,'s',color='DarkOrange')
plt.plot(iterations_markers,l13_markers,'gp',markersize=7)

plt.savefig(name + '_l1.png')
plt.show()

# Dot l1
plt.plot(iterations,dotl11,'b')
plt.plot(iterations,dotl12,color='DarkOrange')
plt.plot(iterations,dotl13,'g')
plt.grid(True)
plt.legend(['Pure exploitation','Epsilon - 0.5','Pure exploration'])
plt.xlabel('Rounds')
plt.ylabel('Relative error')
plt.title('Relative error vs rounds')

# plot markers
plt.plot(iterations_markers,dotl11_markers,'bo')
plt.plot(iterations_markers,dotl12_markers,'s',color='DarkOrange')
plt.plot(iterations_markers,dotl13_markers,'gp',markersize=7)

plt.savefig(name + '_dotl1.png')
plt.show()


# PLOT REGRET

print (iterations_markers)
print (randomregret_markers)

#randomregret_markers = randomregret_markers[0:len(iterations_markers)]
#highdegregret_markers = highdegregret_markers[0:len(iterations_markers)]

print (highdegregret_markers)
# iterations = iterations[0:len(randomregret)]
# randomregret = randomregret[0:len(iterations)]
# highdegregret = highdegregret[0:len(iterations)]


plt.figure()
plt.plot(iterations,randomregret,'r')
plt.plot(iterations,highdegregret,color='purple')
plt.plot(iterations,regret1,'b')
plt.plot(iterations,regret2,color='DarkOrange')
plt.plot(iterations,regret3,'g')
plt.grid(True)
plt.legend(['Random', 'High Degree','Pure exploitation','Epsilon - 0.5','Pure exploration'])
plt.xlabel('Rounds')
plt.ylabel('Average regret')
plt.title('Average regret vs rounds')

# plot markers
plt.plot(iterations_markers_baseline,randomregret_markers,'r*',markersize=8)
plt.plot(iterations_markers_baseline,highdegregret_markers,'x',color='purple',markersize=6)
plt.plot(iterations_markers,regret1_markers,'bo')
plt.plot(iterations_markers,regret2_markers,'s',color='DarkOrange')
plt.plot(iterations_markers,regret3_markers,'gp',markersize=7)

plt.savefig(name + '_regret.png')
plt.show()


