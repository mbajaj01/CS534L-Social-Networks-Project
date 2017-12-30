#define HEAD_INFO

#define HEAD_INFO
//#define HEAD_TRACE
#include "TIM/src/tim.cpp"
#include <random>
#include <iterator>
#include <algorithm>

TimGraph generateGraph(string dataset, string graph_file, int budget, double epsilon=0.1){
	cout<<"Generating Graph"<<endl;
	TimGraph graph(dataset, graph_file);
	cout<<"Finish Read Graph, Start Influecne Maximization"<<endl;
	graph.setInfuModel(InfGraph::IC);
	graph.k = budget;
	return graph;
}

class IC{
	public:
		vector<bool> isInfluenced;
		map<int, map<int, int>> influencedNeighbours;
		vector<int> influenceTimestep;
		random_device rd;
		int graphSize;
		void initAllitems(){
			isInfluenced = vector<bool>(graphSize,false);
			influencedNeighbours.clear();
			influenceTimestep = vector<int>(graphSize,-1);
		}

		int diffusion(TimGraph& graph, vector<int> seeds, bool isEnvironment=true){
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
							dot = graph.prob[currentNode.first][j];
						}else{
							dot = graph.probEstimate[currentNode.first][j];
						}
						double randomProb = (double)dist(rd);
						if(dot > randomProb){
							activeNodes.push_back(pair<int,int>(node, currentNode.second + 1));
							isInfluenced[node] = true;
							influenceTimestep[node] = currentNode.second + 1;
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



		double expectedSpread(TimGraph& graph, vector<int> seeds, int numberOfSimulations, bool isEnvironment){
			double expSpread = 0.0;
			for(int i=0;i<numberOfSimulations;i++){
				expSpread += diffusion(graph, seeds, isEnvironment);
			}
			expSpread /= numberOfSimulations;
			return expSpread;
		}
};


class MAB{
	public:
		int budget;
		IC ic;
		random_device rd;


		MAB(int k){
			budget = k;
			// ic = new IC();
		}

		vector<int> explore(TimGraph& graph){
			int arr[graph.g.size()];
			for(int i = 0; i < graph.g.size(); i++){
			    arr[i] = i;
			}
			random_shuffle(arr, arr+graph.g.size());
			vector<int> seeds;
			for(int i = 0; i < budget; i++){
				seeds.push_back(arr[i]);
			}
			return seeds;
		}

		vector<int> exploit(TimGraph& graph, bool isEnvironment, double epsilon=0.1){
			vector<int> seeds;
			graph.k=budget;
		    graph.setInfuModel(InfGraph::IC);
		    graph.isEnvironment = isEnvironment;
		    graph.EstimateOPT(epsilon);
		    for(auto item:graph.seedSet){
		    	seeds.push_back(item);
		    }
		    return seeds;
		}

		vector<int> epsilonGreedy(TimGraph& graph, double epsilon, double randomNumber){
			vector<int> seeds; 
			if(epsilon > randomNumber){
				seeds = explore(graph);
			}else{
				seeds = exploit(graph, false);
			}
			int spread = ic.diffusion(graph, seeds, true);
			return seeds;
		}

		double L2Error(TimGraph& graph){
			double L2Error = 0.0;
			for (int i=0;i<graph.n;i++){
				for (int j=0;j<graph.g[i].size();j++){
					L2Error += (pow((graph.prob[i][j] - graph.probEstimate[i][j]),2.0));
				}
			}
			return L2Error;
		}

		void learner(TimGraph graph, double epsilon, int iterations){
			mt19937 mt(rd());
		    uniform_real_distribution<double> dist(0.0, 1.0);
		    vector<int> banditSeeds;
		    map<pair<int,int>, int> S_w_plus;
		    map<pair<int,int>, int> S_w_minus;
		    map<pair<int,int>, bool> S_w;
		 //    vector<int> seeds1 = exploit(graph,true);
		 //    int spread1 = ic.expectedSpread(graph, seeds1, 10, true);
			// cout << " " << spread1<<endl;
			for(int t=0; t<iterations;t++){
				double randomNumber = (double)dist(mt);
				banditSeeds = epsilonGreedy(graph, epsilon, randomNumber);
				map<int, map<int, int>>::iterator nodeIterator;
				map<int,double> P_w;
				map<pair<int,int>, bool> isPositiveParent;
				for(nodeIterator=ic.influencedNeighbours.begin(); nodeIterator != ic.influencedNeighbours.end(); nodeIterator++){
					int node = nodeIterator->first;
					long double prob = 1.0;
					map<int, int>::iterator parentIterator;
					for(parentIterator=ic.influencedNeighbours[node].begin(); parentIterator != ic.influencedNeighbours[node].end(); parentIterator++){
						int parent = parentIterator->first;
						int j = graph.edgeMapping[parent][node];
						pair<int, int> edge = pair<int,int>(parent,node);
						isPositiveParent[edge] = false;
						if(S_w_plus.find(edge) == S_w_plus.end()){
							S_w_plus[edge] = 0.0;
							S_w_minus[edge] = 0.0;
							S_w[edge] = true;
						}
						if(ic.influenceTimestep[node] == ic.influenceTimestep[parent] + 1){
							prob *= 1.0 - graph.probEstimate[parent][j];
							S_w_plus[edge] += 1.0;
							isPositiveParent[edge] = true;
						}else{
							S_w_minus[edge] += 1.0;
						}
					}
					if (ic.isInfluenced[node] && ic.influenceTimestep[node] > 0){
						// if (prob == 0.0){
						// 	cout << "Error"<<prob<<endl;
						// }
						P_w[node] = 1.0 - prob;
						if (P_w[node] == 0.0){
							P_w[node] = 1.0;
						}

					}
				}
				

				map<pair<int,int>, bool>::iterator edgeIterator;
				pair<int,int> key;
				for(edgeIterator=isPositiveParent.begin(); edgeIterator != isPositiveParent.end(); edgeIterator++){
					pair<int,int> edge = edgeIterator->first;
					if(S_w.find(edge) != S_w.end()){
						double denominator = S_w_plus[edge] + S_w_minus[edge];
						double prob = 0.0;
						int parent = edge.first;
						int node = graph.edgeMapping[edge.first][edge.second];
						if(isPositiveParent[edge]){
							prob = (graph.probEstimate[parent][node])/(P_w[edge.second]);
							if(prob > 1.0001){
								cout << "Error " << prob << " " << S_w_plus[edge] << " " << S_w_minus[edge]<< endl;
							}
							// if(prob > 1){
							// 	prob = 1.0;
							// }
							
						}
						graph.probEstimate[parent][node] = graph.probEstimate[parent][node] + ((prob - graph.probEstimate[parent][node])/(denominator));
						if(graph.probEstimate[parent][node] == 0.0){
							if(S_w_plus[edge] > 0.0){
								graph.probEstimate[parent][node] = S_w_plus[edge]/denominator;
							}
						}
						graph.probTEstimate[edge.second][graph.edgeMapping[edge.second][edge.first]] = graph.probEstimate[parent][node];
						key = edge;
					}
				}
				cout << L2Error(graph) << " "<< graph.probEstimate[key.first][graph.edgeMapping[key.first][key.second]] << " " << graph.prob[key.first][graph.edgeMapping[key.first][key.second]] << endl;
				
			}

		}

};


int main(int argn, char ** argv)
{
	string dataset = "TIM/nethept/";
	string graph_file = dataset + "graph_ic.inf";
	TimGraph graph = generateGraph(dataset, graph_file, 10);
	// TIC tic();
	// vector<int> seeds;
	// seeds.push_back(1);
	// seeds.push_back(2);
	// tic.expectedSpread(graph, seeds, 100, true);
	MAB mab(50);
	mab.learner(graph, 1, 20000);

}