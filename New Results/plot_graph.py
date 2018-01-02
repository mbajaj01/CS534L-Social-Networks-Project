import matplotlib.pyplot as plt

randomregret = []
highdegregret = []

def read_baselines_random():
	global randomregret
	global highdegregret
	file1='synth_highDegree_50seeds_1000iter'
	file2='synth_random_50seeds_1000iter'
	strdata1 = open(file1, 'rb').read().split('\n')
	strdata2 = open(file2, 'rb').read().split('\n')
	size = len(strdata1)

	regret1 = [] 
	regret2 = []
	idregret = 1
	iterations = []
	for i,datum in enumerate(strdata1):
		if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
			try:
				regret1.append(float(strdata1[i].split(',')[idregret]))
				regret2.append(float(strdata2[i].split(',')[idregret]))
			except:
				print 'could not parse this -> ' + strdata1[i] + strdata2[i]

	print len(regret1)
	print len(regret2)
	highdegregret = regret1
	randomregret = regret2


def read_baselines_nethept():
	global randomregret
	global highdegregret
	file1='nethept_highdegree_50seeds_1000iter'
	file2='nethept_random_50seeds_1000iter'
	strdata1 = open(file1, 'rb').read().split('\n')
	strdata2 = open(file2, 'rb').read().split('\n')
	size = len(strdata1)

	regret1 = [] 
	regret2 = []
	idregret = 1
	iterations = []
	for i,datum in enumerate(strdata1):
		if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
			try:
				regret1.append(float(strdata1[i].split(',')[idregret]))
				regret2.append(float(strdata2[i].split(',')[idregret]))
			except:
				print 'could not parse this -> ' + strdata1[i] + strdata2[i]

	print len(regret1)
	print len(regret2)
	highdegregret = regret1
	randomregret = regret2


def plot_curves2(file1,file2,file3,name):
	global randomregret
	global highdegregret
	strdata1 = open(file1, 'rb').read().split('\n')
	strdata2 = open(file2, 'rb').read().split('\n')
	strdata3 = open(file3, 'rb').read().split('\n')
	size = len(strdata1)

	l11 = []
	l12 = []
	l13 = []
	idl1 = 1
	regret1 = [] 
	regret2 = []
	regret3 = []
	idregret = 2
	iterations = []
	for i,datum in enumerate(strdata1):
		if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
			try:
				l11.append(float(strdata1[i].split(' ')[idl1]))
				l12.append(float(strdata2[i].split(' ')[idl1]))
				l13.append(float(strdata3[i].split(' ')[idl1]))
				regret1.append(float(strdata1[i].split(' ')[idregret]))
				regret2.append(float(strdata2[i].split(' ')[idregret]))
				regret3.append(float(strdata3[i].split(' ')[idregret]))
				iterations.append(i+1)
			except:
				print 'could not parse this -> ' + strdata1[i] + strdata2[i] + strdata3[i] 

	print len(l11)
	print len(l12)
	print len(l13)
	print len(regret1)
	print len(regret2)
	print len(regret3)

	print iterations
	print l11

	randomregret = randomregret[0:len(iterations)]
	highdegregret = highdegregret[0:len(iterations)]

	plt.plot(iterations,l11,'bo-')
	plt.plot(iterations,l12,'s-',color='DarkOrange')
	plt.plot(iterations,l13,'gx-')
	plt.grid(True)
	plt.legend(['Pure exploitation','Epsilon - 0.5','Pure exploration'])
	plt.xlabel('Rounds')
	plt.ylabel('Relative error')
	plt.title('Relative error vs rounds')
	plt.savefig(name + '_l1.png')
	plt.show()

	plt.figure()
	plt.plot(iterations,randomregret,'r*-')
	plt.plot(iterations,highdegregret,'2-',color='purple')
	plt.plot(iterations,regret1,'bo-')
	plt.plot(iterations,regret2,'s-',color='DarkOrange')
	plt.plot(iterations,regret3,'gx-')
	plt.grid(True)
	plt.legend(['Random', 'High Degree','Pure exploitation','Epsilon - 0.5','Pure exploration'])
	plt.xlabel('Rounds')
	plt.ylabel('Average regret')
	plt.title('Average regret vs rounds')
	plt.savefig(name + '_regret.png')
	plt.show()

def plot_curves(file1,file2,file3,name):
	global randomregret
	global highdegregret
	strdata1 = open(file1, 'rb').read().split('\n')
	strdata2 = open(file2, 'rb').read().split('\n')
	strdata3 = open(file3, 'rb').read().split('\n')
	size = len(strdata1)

	l11 = []
	l12 = []
	l13 = []
	idl1 = 0
	regret1 = [] 
	regret2 = []
	regret3 = []
	idregret = 1
	iterations = []
	for i,datum in enumerate(strdata1):
		if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
			try:
				l11.append(float(strdata1[i].split(',')[idl1]))
				l12.append(float(strdata2[i].split(',')[idl1]))
				l13.append(float(strdata3[i].split(',')[idl1]))
				regret1.append(float(strdata1[i].split(',')[idregret]))
				regret2.append(float(strdata2[i].split(',')[idregret]))
				regret3.append(float(strdata3[i].split(',')[idregret]))
				iterations.append(i+1)
			except:
				print 'could not parse this -> ' + strdata1[i] + strdata2[i] + strdata3[i] 

	print len(l11)
	print len(l12)
	print len(l13)
	print len(regret1)
	print len(regret2)
	print len(regret3)
	randomregret = randomregret[0:len(iterations)]
	highdegregret = highdegregret[0:len(iterations)]
	iterations = iterations[0:len(randomregret)]
	l11 = l11[0:len(randomregret)]
	l12 = l12[0:len(randomregret)]
	l13 = l13[0:len(randomregret)]
	regret1 = regret1[0:len(randomregret)]
	regret2 = regret2[0:len(randomregret)]
	regret3 = regret3[0:len(randomregret)]


	plt.plot(iterations,l11,'o-')
	plt.plot(iterations,l12,'s-')
	plt.plot(iterations,l13,'x-')
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation'])
	plt.xlabel('Rounds')
	plt.ylabel('Relative error')
	plt.title('Relative error vs rounds')
	plt.savefig(name + '_l1.png')
	plt.show()

	plt.figure()
	plt.plot(iterations,regret1,'o-')
	plt.plot(iterations,regret2,'s-')
	plt.plot(iterations,regret3,'x-')
	plt.plot(iterations,randomregret,'*-')
	plt.plot(iterations,highdegregret,'2-')
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation','Random', 'High Degree'])
	plt.xlabel('Rounds')
	plt.ylabel('Average regret')
	plt.title('Average regret vs rounds')
	plt.savefig(name + '_regret.png')
	plt.show()

def plot_curves_tic(file1,file2,file3,name):
	global randomregret
	global highdegregret
	strdata1 = open(file1, 'rb').read().split('\n')
	strdata2 = open(file2, 'rb').read().split('\n')
	strdata3 = open(file3, 'rb').read().split('\n')
	size = len(strdata1)

	l11 = []
	l12 = []
	l13 = []
	idl1 = 0
	rell11 = []
	rell12 = []
	rell13 = []
	idrell1 = 1
	regret1 = [] 
	regret2 = []
	regret3 = []
	idregret = 2
	iterations = []
	for i,datum in enumerate(strdata1):
		if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
			try:
				l11.append(float(strdata1[i].split(',')[idl1]))
				l12.append(float(strdata2[i].split(',')[idl1]))
				l13.append(float(strdata3[i].split(',')[idl1]))
				rell11.append(float(strdata1[i].split(',')[idrell1]))
				rell12.append(float(strdata2[i].split(',')[idrell1]))
				rell13.append(float(strdata3[i].split(',')[idrell1]))
				regret1.append(float(strdata1[i].split(',')[idregret]))
				regret2.append(float(strdata2[i].split(',')[idregret]))
				regret3.append(float(strdata3[i].split(',')[idregret]))
				iterations.append(i+1)
			except:
				print 'could not parse this -> ' + strdata1[i] + strdata2[i] + strdata3[i] 

	print len(l11)
	print len(l12)
	print len(l13)
	print len(rell11)
	print len(rell12)
	print len(rell13)
	print len(regret1)
	print len(regret2)
	print len(regret3)
	randomregret = randomregret[0:len(iterations)]
	highdegregret = highdegregret[0:len(iterations)]
	iterations = iterations[0:len(randomregret)]
	l11 = l11[0:len(randomregret)]
	l12 = l12[0:len(randomregret)]
	l13 = l13[0:len(randomregret)]
	rell11 = rell11[0:len(randomregret)]
	rell12 = rell12[0:len(randomregret)]
	rell13 = rell13[0:len(randomregret)]
	regret1 = regret1[0:len(randomregret)]
	regret2 = regret2[0:len(randomregret)]
	regret3 = regret3[0:len(randomregret)]


	plt.plot(iterations,l11,'o-')
	plt.plot(iterations,l12,'s-')
	plt.plot(iterations,l13,'x-')
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation'])
	plt.xlabel('Rounds')
	plt.ylabel('Relative error')
	plt.title('Relative error vs rounds')
	plt.savefig(name + '_l1.png')
	plt.show()

	plt.plot(iterations,rell11,'o-')
	plt.plot(iterations,rell12,'s-')
	plt.plot(iterations,rell13,'x-')
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation'])
	plt.xlabel('Rounds')
	plt.ylabel('Relative error (dot)')
	plt.title('Relative error (dot) vs rounds')
	plt.savefig(name + '_l1_dot.png')
	plt.show()

	plt.figure()
	plt.plot(iterations,regret1,'o-')
	plt.plot(iterations,regret2,'s-')
	plt.plot(iterations,regret3,'x-')
	plt.plot(iterations,randomregret,'*-')
	plt.plot(iterations,highdegregret,'2-')
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation','Random', 'High Degree'])
	plt.xlabel('Rounds')
	plt.ylabel('Average regret')
	plt.title('Average regret vs rounds')
	plt.savefig(name + '_regret.png')
	plt.show()

read_baselines_random()
print len(randomregret)
print len(highdegregret)

# Edge level synth
file1 = 'log_synth_edge_0.txt'
file2 = 'log_synth_edge_0.5.txt'
file3 = 'log_synth_edge_1.txt'
#plot_curves(file1,file2,file3,'synth_edge')

# Node level synth
file1 = 'log_0.000000synth_5000_l.txt'
file2 = 'log_0.500000synth_5000_l.txt'
file3 = 'log_1.000000synth_5000_l.txt'
#plot_curves2(file1,file2,file3,'synth_node')

read_baselines_nethept()
# Node level nethept
file1 = 'log_0.000000graph_ic.inf' # seems like mohit has named the files wrong
file2 = 'log_0.500000graph_ic.inf' # so I have swapped his explore and exploit
file3 = 'log_1.000000graph_ic.inf'
plot_curves2(file1,file2,file3,'nethept_node')

#Edge level nethelp
file1 = 'nethept_edge_PExplore_50seeds_1000iter'
file2 = 'nethept_edge_EGreedy_50seeds_1000iter'
file3 = 'nethept_edge_PExp_50seeds_1000iter'
#plot_curves(file1,file2,file3,'nethept_edge')

# TIC CURVES

# Edge level nethelp 10items 10topics
file1 = 'nethept_tic_edge_Explore_50seeds_1000iter_10items_10topics'
file2 = 'nethept_tic_edge_EGreedy_50seeds_1000iter_10items_10topics'
file3 = 'nethept_tic_edge_Exploit_50seeds_1000iter_10items_10topics'
#plot_curves_tic(file1,file2,file3,'tic_nethept_edge_10_10')


# Edge level nethelp 10items 5topics
file1 = 'nethept_tic_edge_Explore_50seeds_1000iter_10items_5topics'
file3 = 'nethept_tic_edge_EGreedy_50seeds_1000iter_10items_10topics'
file4 = 'nethept_tic_edge_Exploit_50seeds_1000iter_10items_10topics'
#plot_curves_tic(file1,file2,file3,'tic_nethept_edge_10_5')

# Edge level nethelp 5items 10topics
file1 = 'nethept_tic_edge_Explore_50seeds_1000iter_5items_10topics'
file2 = 'nethept_tic_edge_EGreedy_50seeds_1000iter_10items_10topics'
file3 = 'nethept_tic_edge_Exploit_50seeds_1000iter_10items_10topics'
#plot_curves_tic(file1,file2,file3,'tic_nethept_edge_5_10')


# Node level nethelp 10items 10topics
file1 = 'nethept_tic_node_Explore_50seeds_1000iter_10items_10topics'
file2 = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_10topics'
file3 = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_10topics'
#plot_curves_tic(file1,file2,file3,'tic_nethept_node_10_10')


# Node level nethelp 10items 5topics
file1 = 'nethept_tic_node_Explore_50seeds_1000iter_10items_5topics'
file2 = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_10topics'
file3 = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_10topics'
#plot_curves_tic(file1,file2,file3,'tic_nethept_node_10_5')

# Node level nethelp 5items 10topics
file1 = 'nethept_tic_node_Explore_50seeds_1000iter_5items_10topics'
file2 = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_10topics'
file3 = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_10topics'
#plot_curves_tic(file1,file2,file3,'tic_nethept_node_5_10')





