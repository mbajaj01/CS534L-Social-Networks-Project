import matplotlib.pyplot as plt


def plot_curves2(file1,file2,file3,name):
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
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation'])
	plt.xlabel('Rounds')
	plt.ylabel('Average regret')
	plt.title('Average regret vs rounds')
	plt.savefig(name + '_regret.png')
	plt.show()


def plot_curves(file1,file2,file3,name):
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
	plt.grid(True)
	plt.legend(['Pure exploration','Epsilon - 0.5','Pure exploitation'])
	plt.xlabel('Rounds')
	plt.ylabel('Average regret')
	plt.title('Average regret vs rounds')
	plt.savefig(name + '_regret.png')
	plt.show()


# Edge level synth
file1 = 'log_synth_edge_0.txt'
file2 = 'log_synth_edge_0.5.txt'
file3 = 'log_synth_edge_1.txt'
plot_curves(file1,file2,file3,'synth_edge')

# Node level synth
file1 = 'log_0.000000synth_5000_l.txt'
file2 = 'log_0.500000synth_5000_l.txt'
file3 = 'log_1.000000synth_5000_l.txt'
plot_curves2(file1,file2,file3,'synth_node')

# Node level nethept
file1 = 'log_0.000000graph_ic.inf'
file2 = 'log_0.500000graph_ic.inf'
file3 = 'log_1.000000graph_ic.inf'
plot_curves2(file1,file2,file3,'nethept_node')


