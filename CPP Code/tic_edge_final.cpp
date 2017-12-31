#define HEAD_INFO

#define HEAD_INFO
//#define HEAD_TRACE
#include "TIM_TIC/src/tim.cpp"
#include <random>
#include <iterator>
#include <algorithm>

TimGraph generateGraph(string dataset, string graph_file, int budget, int numTopics, double epsilon=0.1){
	cout<<"Generating Graph"<<endl;
	TimGraph graph(dataset, graph_file, numTopics);
	cout<<"Finish Read Graph, Start Influecne Maximization"<<endl;
	graph.setInfuModel(InfGraph::IC);
	graph.k = budget;
	return graph;
}

class Item{
	public:
		vector<double> estimates;
		vector<double> trueDistribution;
		vector<double> logEstimates;

	void normalizeEstimates(){
		double sum = 0.0;
		for (int i=0; i<(int)estimates.size(); i++){
			sum += estimates[i];
		}
		for (int i=0; i<(int)estimates.size(); i++){
			estimates[i] = estimates[i]/sum;
		}
	}

	void normalizeTrueDistribution(){
		double sum = 0.0;
		for (int i=0; i<(int)trueDistribution.size(); i++){
			sum += trueDistribution[i];
		}
		for (int i=0; i<(int)trueDistribution.size(); i++){
			trueDistribution[i] /= sum;
		}
	}
};



class TIC{
	public:
		vector<bool> isInfluenced;
		map<int, map<int, int>> influencedNeighbours;
		vector<int> influenceTimestep;
		map<int, int> influencingEdge;
		random_device rd;
		int graphSize;
		void initAllitems(){
			isInfluenced = vector<bool>(graphSize,false);
			influencedNeighbours.clear();
			influenceTimestep = vector<int>(graphSize,-1);
			influencingEdge.clear();
		}

		double dotProductFunction(vector<double> vec1, vector<double> vec2){
            double dotprod = 0.0;
            for (int i=0;i<(int)vec1.size();i++){
                dotprod += vec1[i]*vec2[i];
            }
            return dotprod;
        }

		int diffusion(TimGraph& graph, vector<double> item, vector<int> seeds, bool isEnvironment=true){
			graphSize = graph.n;
			initAllitems();
			deque<pair<int,int>>activeNodes;
			int spread = 0;
		    mt19937 mt(rd());
		    uniform_real_distribution<double> dist(0.0, 1.0);
			for(int i=0;i<(int)seeds.size();i++){
				activeNodes.push_back(pair<int,int>(seeds[i],0));
				isInfluenced[seeds[i]] = true;
				influenceTimestep[seeds[i]] = 0;
				spread += 1;
			}
			while(!activeNodes.empty()){
				pair<int,int> currentNode = activeNodes.front();
				activeNodes.pop_front();
				for(int j=0; j<(int)graph.g[currentNode.first].size(); j++){
					int node = graph.g[currentNode.first][j];
					if(!isInfluenced[node]){
						map<int, map<int, int>>::iterator influencedNeighboursIt;
						influencedNeighboursIt = influencedNeighbours.find(node);
						if(influencedNeighboursIt == influencedNeighbours.end()){
							influencedNeighbours[node] = map<int, int>();
						}
						influencedNeighbours[node][currentNode.first] = currentNode.second;
						double dot;
						if(isEnvironment){
							dot = dotProductFunction(item, graph.prob[currentNode.first][j]);
						}else{
							dot = dotProductFunction(item, graph.probEstimate[currentNode.first][j]);
						}
						double randomProb = (double)dist(rd);
						if(dot > randomProb){
							activeNodes.push_back(pair<int,int>(node, currentNode.second + 1));
							isInfluenced[node] = true;
							influenceTimestep[node] = currentNode.second + 1;
							influencingEdge[node] = currentNode.first;
							spread += 1;
						}
						
					}else{
						if(currentNode.second < influenceTimestep[node]){
							influencedNeighbours[node][currentNode.first] = currentNode.second;
						}
					}
				}
			}
			return spread;
		}

		double compareRegret(TimGraph& graph, vector<double> item, vector<int> trueSeeds, vector<int> banditSeeds){
			vector<vector<int>> possibleWorld;
			mt19937 mt(rd());
		    uniform_real_distribution<double> dist(0.0, 1.0);
			for(int i=0; i<graph.n; i++){
				int parent = i;
				possibleWorld.push_back(vector<int>());
				for(int j=0; j<(int)graph.g[i].size(); j++){
					int node = graph.g[i][j];
					double randomProb = (double)dist(rd);
					if (dotProductFunction(item, graph.prob[parent][j]) > randomProb){
						possibleWorld[parent].push_back(1);
					}else{
						possibleWorld[parent].push_back(0);
					}
				}
			}
			
			double spread = 0;
			deque<pair<int,int>>activeNodes;
			vector<bool> visited(graph.n, false);
			for(int i=0;i<(int)trueSeeds.size();i++){
				activeNodes.push_back(pair<int,int>(trueSeeds[i],0));
				visited[trueSeeds[i]] = true;
				spread += 1;
			}
			while(!activeNodes.empty()){
				pair<int,int> currentNode = activeNodes.front();
				activeNodes.pop_front();
				for(int j=0; j<(int)graph.g[currentNode.first].size(); j++){
					int node = graph.g[currentNode.first][j];
					if(!visited[node]){
						if(possibleWorld[currentNode.first][j] == 1){
							activeNodes.push_back(pair<int,int>(node, currentNode.second + 1));
							spread += 1;
							visited[node] = true;
						}
					}
				}
			}

			activeNodes.clear();
			visited = vector<bool>(graph.n, false);
			double spreadBandit = 0;
			for(int i=0;i<(int)banditSeeds.size();i++){
				activeNodes.push_back(pair<int,int>(banditSeeds[i],0));
				visited[banditSeeds[i]] = true;
				spreadBandit += 1;
			}
			while(!activeNodes.empty()){
				pair<int,int> currentNode = activeNodes.front();
				activeNodes.pop_front();
				for(int j=0; j<(int)graph.g[currentNode.first].size(); j++){
					int node = graph.g[currentNode.first][j];
					if(!visited[node]){
						if(possibleWorld[currentNode.first][j] == 1){
							activeNodes.push_back(pair<int,int>(node, currentNode.second + 1));
							spreadBandit += 1;
							visited[node] = true;
						}
					}
				}
			}
			// cout << spread << " " << spreadBandit << endl;
			return (spread - spreadBandit);
		}

		double expectedSpread(TimGraph& graph, vector<double> item, vector<int> seeds, int numberOfSimulations, bool isEnvironment){
			double expSpread = 0.0;
			for(int i=0;i<numberOfSimulations;i++){
				expSpread += diffusion(graph, item, seeds, isEnvironment);
			}
			expSpread /= numberOfSimulations;
			return expSpread;
		}
};


class MAB{
	public:
		int budget, numTopics, numItems;
		TIC tic;
		random_device rd;


		MAB(int k, int topics, int itemNumber){
			budget = k;
			numTopics = topics;
			numItems = itemNumber;
			// ic = new IC();
		}

		vector<int> explore(TimGraph& graph){
			int arr[graph.g.size()];
			for(int i = 0; i < (int)graph.g.size(); i++){
			    arr[i] = i;
			}
			random_shuffle(arr, arr+graph.g.size());
			vector<int> seeds;
			for(int i = 0; i < budget; i++){
				seeds.push_back(arr[i]);
			}
			return seeds;
		}

		vector<int> exploit(TimGraph& graph, Item item, bool isEnvironment, double epsilon=0.1){
			vector<int> seeds;
			graph.k=budget;
		    graph.setInfuModel(InfGraph::IC);
		    graph.isEnvironment = isEnvironment;
		    if(isEnvironment){
		    	graph.itemDistribution = item.trueDistribution;
		    }else{
		    	graph.itemDistribution = item.estimates;
		    }
		    graph.EstimateOPT(epsilon);
		    for(auto item:graph.seedSet){
		    	seeds.push_back(item);
		    }
		    return seeds;
		}

		vector<int> epsilonGreedy(TimGraph& graph, Item item, double epsilon, double randomNumber, vector<int> banditSeeds){
			vector<int> seeds; 
			if(epsilon > randomNumber){
				seeds = explore(graph);
			}else{
				// seeds = exploit(graph, false);
				seeds = banditSeeds;
			}
			int spread = tic.diffusion(graph, item.trueDistribution, seeds, true);
			return seeds;
		}

		double L2Error(TimGraph& graph){
			double L2Error = 0.0;
			for (int i=0;i<graph.n;i++){
				for (int j=0;j<(int)graph.g[i].size();j++){
					for (int k=0; k<(int)graph.prob[i][j].size();k++){
						L2Error += (pow((graph.prob[i][j][k] - graph.probEstimate[i][j][k]),2.0));
					}
				}
			}
			return L2Error;
		}

		double L2DotError(TimGraph& graph, Item item){
			double L2Error = 0.0;
			for (int i=0;i<graph.n;i++){
				for (int j=0;j<(int)graph.g[i].size();j++){
					double diff = tic.dotProductFunction(graph.prob[i][j], item.trueDistribution) - tic.dotProductFunction(graph.probEstimate[i][j], item.estimates);
					L2Error += pow(diff, 2.0);
				}
			}
			return L2Error;
		}

		double logSumExp(vector<double> vec){
			double maxVal = vec[0];
			double sum = 0.0;
			for(int i=1; i<(int)vec.size(); i++){
				if(vec[i] > maxVal){
					maxVal = vec[i];
				}
			}

			for(int i=0; i<(int)vec.size(); i++){
				sum += exp(vec[i] - maxVal);
			}
			return (log(sum) + maxVal);
		}

		void learner(TimGraph graph, vector<Item> items, double epsilon, int iterations){
			mt19937 mt(rd());
		    uniform_real_distribution<double> dist(0.0, 1.0);
		    vector<vector<int>> banditSeeds;
		    vector<vector<int>> trueSeeds;
		   	Item pi;
		   	vector<double> e;
		   	map<pair<int,int>,vector<double>> negativeSum;
		   	map<pair<int,int>,vector<double>> positiveSum;
			for (int j=0; j< numTopics; j++){
				double randomNumberE = dist(mt);
				e.push_back(randomNumberE);
			}
			pi.estimates = e;
			pi.logEstimates = vector<double>(numTopics, 0.0);
			pi.normalizeEstimates();
			for (int j=0; j< numTopics; j++){
				pi.logEstimates[j] = log(pi.estimates[j]);
			}
		   	for(int item=0; item<(int)items.size(); item++){
		   		vector<int> trueSeed = exploit(graph,items[item],true);
		   		trueSeeds.push_back(trueSeed);
		   		cout << tic.expectedSpread(graph, items[0].trueDistribution, trueSeeds[item], 10 ,true) << endl;
		   		banditSeeds.push_back(explore(graph));
		   	}
		   	vector<double> piIteration(numTopics, 0.0);
		   	for (int t=0; t<iterations; t++){
		   		//Define Variables
				vector<vector<double>> Q;
				map<pair<int,int>,vector<int>> SPlus;
				map<pair<int,int>,vector<int>> SMinus;
				map<pair<int,int>,bool> kappa;

				//Iterate over all items
				for (int item=0; item<(int)items.size(); item++){
					double randomNumber = (double)dist(mt);
					banditSeeds[item] = epsilonGreedy(graph, items[item], epsilon, randomNumber, banditSeeds[item]);
					vector<double> cascadeLogProbPositive(numTopics,0.0);
					vector<double> cascadeLogProbNegative(numTopics,0.0);
					map<int, map<int, int>>::iterator nodeIterator;
					Q.push_back(vector<double>(numTopics, 0.0));

					//Iterate over all nodes that have some influenced parent
					for(nodeIterator=tic.influencedNeighbours.begin(); nodeIterator != tic.influencedNeighbours.end(); nodeIterator++){
						int node = nodeIterator->first;
						map<int, int>::iterator parentIterator;
						//Iterate over all parents of that node
						for(parentIterator=tic.influencedNeighbours[node].begin(); parentIterator != tic.influencedNeighbours[node].end(); parentIterator++){
							int parent = parentIterator->first;
							int j = graph.edgeMapping[parent][node];
							pair<int, int> edge = pair<int,int>(parent,node);
							kappa[edge] = true;

							//Check if (parent, node) is th.e activating edge
							if(tic.isInfluenced[node]){
								if(tic.influenceTimestep[node] > 0){
									if(tic.influencingEdge[node] == parent){
										for(int z=0; z<numTopics; z++){
											double positive =  graph.probEstimate[parent][j][z];
											if (positive <= 0.0){
												positive = 1.0;
											}
											cascadeLogProbPositive[z] += log(positive);
										}
										if (SPlus.find(edge) == SPlus.end()){
											SPlus[edge] = vector<int>();
										}
										SPlus[edge].push_back(item);
									}else{
										for(int z=0; z<numTopics; z++){
											double negative = 1.0 - graph.probEstimate[parent][j][z];
											if (negative <= 0.0){
												negative = 1.0;
											}
											cascadeLogProbNegative[z] += log(negative);
										}
										if (SMinus.find(edge) == SMinus.end()){
											SMinus[edge] = vector<int>();
										}
										SMinus[edge].push_back(item);
									}
								}
							}else{
								for(int z=0; z<numTopics; z++){
									double negative = 1.0 - graph.probEstimate[parent][j][z];
									if (negative <= 0.0){
										negative = 1.0;
									}
									cascadeLogProbNegative[z] += log(negative);
								}
								if (SMinus.find(edge) == SMinus.end()){
									SMinus[edge] = vector<int>();
								}
								SMinus[edge].push_back(item);
							}
						}
					}

					//Update item estimate
					for(int z=0; z<numTopics; z++){
						Q[item][z] = pi.logEstimates[z] + cascadeLogProbPositive[z] + cascadeLogProbNegative[z];
						if(Q[item][z] < -100){
							Q[item][z] = -100;
						}
					}

					//Normalize
					double norm = logSumExp(Q[item]);
					double sum = 0.0;
					for(int z=0; z<numTopics; z++){
						double number = exp(Q[item][z] - norm);
						Q[item][z] = number;
						piIteration[z] += number;
						// cout << "Q Value: "<< Q[item][z] << endl;
						items[item].estimates[z] += (Q[item][z] - items[item].estimates[z])/(t+1);
						sum += items[item].estimates[z];
					}
					// if (sum != 1.0){
					// 	cout << "Item Estimate don't sum to 1: "<< sum << endl;
					// }
				}

				//Update Average Topic
				for(int z=0; z<numTopics; z++){
					piIteration[z] /= numItems;
					pi.estimates[z] += (piIteration[z] - pi.estimates[z])/(t+1);
					pi.logEstimates[z] = log(pi.estimates[z]);
					// cout << "Log Estimate: "<< pi.logEstimates[z] << " " << pi.estimates[z] << endl;
				}


				//Update Edge Probabilities
				map<pair<int,int>, bool>::iterator edgeIterator;
				pair<int,int> key;
				for(edgeIterator=kappa.begin(); edgeIterator != kappa.end(); edgeIterator++){
					vector<double> numerator(numTopics,0.0);
					vector<double> denominator(numTopics,0.0);
					pair<int, int> edge = edgeIterator->first;
					if(negativeSum.find(edge) == negativeSum.end()){
						negativeSum[edge] = vector<double>(numTopics, 0.0);
					}
					if(positiveSum.find(edge) == positiveSum.end()){
						positiveSum[edge] = vector<double>(numTopics, 0.0);
					}

					bool update = false;
					if(SPlus.find(edge) != SPlus.end()){
						for(int itemIndex =0; itemIndex< (int)SPlus[edge].size(); itemIndex++){
							int item = SPlus[edge][itemIndex];
							for(int z=0; z<numTopics; z++){
								numerator[z] += Q[item][z];
								denominator[z] += Q[item][z];
								if(numerator[z] == 0.0){
									cout << "Numerator 0: "<< numerator[z] << " "<<denominator[z]<< " "<< " "<<Q[item][z] << endl;
									exit(0);
								}
							}
						}
						key = edge;
						update = true;
					}

					if(SMinus.find(edge) != SMinus.end()){
						for(int itemIndex =0; itemIndex< (int)SMinus[edge].size(); itemIndex++){
							int item = SMinus[edge][itemIndex];
							for(int z=0; z<numTopics; z++){
								denominator[z] += Q[item][z];
								if(denominator[z] == 0.0){
									cout << "denominator 0: "<< numerator[z] << " "<<denominator[z]<< " "<< " "<<Q[item][z] << endl;
									exit(0);
								}
							}
						}
						update = true;
					}

					for(int z=0; z<numTopics; z++){
						positiveSum[edge][z] += numerator[z];
						negativeSum[edge][z] += denominator[z];
					}

					if(!update){
						cout << "Something is Wrong" << endl;
						exit(0);
					}

					if(update){
						int parent = edge.first;
						int j = graph.edgeMapping[edge.first][edge.second];
						for(int z=0; z<numTopics; z++){
							if(numerator[z] > denominator[z]){
								cout << "Numerator > Denominator: "<< numerator[z] << " "<<denominator[z] << endl;
								exit(0);
							}
							double prob = numerator[z]/denominator[z];
							if(prob == 0 && numerator[z] != 0){
								cout << "Underflow: "<< prob << " "<<numerator[z] << " "<<denominator[z] << endl;
								exit(0);
							}
							graph.probEstimate[parent][j][z] += ( (prob - graph.probEstimate[parent][j][z]) * (denominator[z]/negativeSum[edge][z]) );
							// graph.probEstimate[parent][node][z] += positiveSum[edge][z]/negativeSum[edge][z];
							if (graph.probEstimate[parent][j][z] == 0.0){
								if (positiveSum[edge][z] > 0.0){
									graph.probEstimate[parent][j][z] = positiveSum[edge][z]/negativeSum[edge][z];
								}
							}
							graph.probTEstimate[edge.second][graph.edgeTMapping[edge.second][edge.first]][z] = graph.probEstimate[parent][j][z];
						}
					}

				}
				int index = graph.edgeMapping[key.first][key.second];
				cout << L2Error(graph) << " " << L2DotError(graph, items[0]) << " " <<key.first<< " "<<key.second<< " " <<tic.dotProductFunction(graph.probEstimate[key.first][index], items[0].estimates)   << " " << tic.dotProductFunction(graph.prob[key.first][index], items[0].trueDistribution) << endl;
			

		   	}

		}

};

int main(int argn, char ** argv)
{
	string dataset = "TIM_TIC/nethept/";
	string graph_file = dataset + "graph_tic_2.inf";
	int numTopics = 2;
	int numItems = 2;
	TimGraph graph = generateGraph(dataset, graph_file, 10, numTopics);
	// for(int i=0; i<graph.n; i++){
	// 	for(int j=0; j<graph.g[i].size(); j++){
	// 		int node1 = i;
	// 		int node2 = graph.g[i][j];
	// 		cout << node1 << " " << node2 << " ";
	// 		for (int z=0; z<numTopics; z++){
	// 			cout << graph.prob[i][j][z] << " ";
	// 		}
	// 		cout << endl;
	// 	}
	// }
	// exit(0);
	vector<Item> items;
	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i=0;i<numItems;i++){
		Item item;
		vector<double> e;
		vector<double> d;
		for (int j=0; j< numTopics; j++){
			double randomNumberE = dist(mt);
			e.push_back(randomNumberE);
			double randomNumberD = dist(mt);
			d.push_back(randomNumberD);
		}
		item.estimates = e;
		item.trueDistribution = d;
		item.normalizeEstimates();
		item.normalizeTrueDistribution();
		items.push_back(item);
	}
	MAB mab(50, numTopics, numItems);
	mab.learner(graph, items, 1,10000);

}