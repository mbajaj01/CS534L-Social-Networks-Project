import matplotlib.pyplot as plt

#TIC node
# NORMALIZE YES
# exploit_file = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_10topics'
# ep_greedy = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_10topics'
explore_file_1010 = 'nethept_tic_node_Explore_50seeds_1000iter_10items_10topics'


# baseline_highdeg='nethept_HighDegree_50seeds_1000iter_10items_5topics'
# baseline_random='nethept_random_50seeds_1000iter_10items_5topics'
# exploit_file = 'nethept_tic_node_Exploit_50seeds_1000iter_10items_5topics'
# ep_greedy = 'nethept_tic_node_EGreedy_50seeds_1000iter_10items_5topics'
explore_file_105 = 'nethept_tic_node_Explore_50seeds_1000iter_10items_5topics'


# NORMALIZE YES
# baseline_highdeg='nethept_HighDegree_50seeds_1000iter_5items_10topics'
# baseline_random='nethept_random_50seeds_1000iter_5items_10topics'
# exploit_file = 'nethept_tic_node_Exploit_50seeds_1000iter_5items_10topics'
# ep_greedy = 'nethept_tic_node_EGreedy_50seeds_1000iter_5items_10topics'
explore_file_510 = 'nethept_tic_node_Explore_50seeds_1000iter_5items_10topics'



str_data_1010 = open(explore_file_1010,'rb').read().split('\n')
str_data_105 = open(explore_file_105,'rb').read().split('\n')
str_data_510 = open(explore_file_510,'rb').read().split('\n')

id_l1 = 0
l1_1010_markers = []
l1_105_markers = []
l1_510_markers = []
l1_1010 = []
l1_105 = []
l1_510 = []
iterations = []
iterations_markers = []

for i,datum in enumerate(str_data_1010):
	if (i%2==0) and (i%3==0) and (i%4==0) and (i%6==0) and (i%8==0) and (i%12==0) and (i%16==0):
		try:
			l1_1010_markers.append(float(str_data_1010[i].split(',')[id_l1]))
			l1_105_markers.append(float(str_data_105[i].split(',')[id_l1]))
			l1_510_markers.append(float(str_data_510[i].split(',')[id_l1]))
			iterations_markers.append(i+1)
		except:
			print 'could not parse this -> ' + datum
			pass
	try:
		l1_1010.append(float(str_data_1010[i].split(',')[id_l1]))
		l1_105.append(float(str_data_105[i].split(',')[id_l1]))
		l1_510.append(float(str_data_510[i].split(',')[id_l1]))
		iterations.append(i+1)
	except:
		print 'could not parse this -> ' + datum
		pass


print len(l1_1010)
print len (l1_1010_markers)

plt.plot(iterations,l1_1010,'b',color="#333745")
plt.plot(iterations,l1_105,color="#e63462")
plt.grid(True)
plt.legend(['10 topics','5 topics'])
plt.xlabel('Rounds')
plt.ylabel('Relative error')
plt.title('Relative error vs rounds for pure exploration')

# plot markers
plt.plot(iterations_markers,l1_1010_markers,'o',color="#333745")
plt.plot(iterations_markers,l1_105_markers,'s',color="#e63462")

plt.savefig('tic_error_topics.png')
plt.show()



plt.plot(iterations,l1_1010,color="#333745")
plt.plot(iterations,l1_510,color="#e63462")
plt.grid(True)
plt.legend(['10 items','5 items'])
plt.xlabel('Rounds')
plt.ylabel('Relative error')
plt.title('Relative error vs rounds for pure exploration')

# plot markers
plt.plot(iterations_markers,l1_1010_markers,'o',color="#333745")
plt.plot(iterations_markers,l1_510_markers,'s',color="#e63462")

plt.savefig('tic_error_items.png')
plt.show()


