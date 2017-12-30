#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <iostream>
#include <assert.h>
//#include <pair>

using namespace std;


struct NodeInfo{
	NodeInfo(int id){
		isInfluenced = false;
		timestep = 0;
		nodeId = id;
		influencedBy = -1;
	}
	int nodeId;
	map<int,int> successors;  // node , edge id
	map<int, int> predecessors; // node, edge id    
	int influencedBy;
	bool isInfluenced;
	int timestep;
};

struct EdgeInfo{
	EdgeInfo(int fromId, int toId, double prob, double est){
		isLive = false;
		fromNode = fromId;
		toNode = toId;
		estimate = est;
		probability = prob;
	}
	bool isLive;
	int fromNode;
	int toNode;
	double estimate;
	double probability;
};


class Xgraph
{
	public:
		Xgraph(){

		};

		void ResetAttributes(){
			for(auto & iter : m_nodeInfo){
				iter.isInfluenced = false;
				iter.timestep = 0;
				iter.influencedBy = -1;
			}

			for(auto & iter : m_edgeInfo){
				iter.isLive = false;
			}
		};

		bool HasNode(int node){
			return m_nodeToIndex.find(node) != m_nodeToIndex.end();
		};

		vector<int> Nodes(){
			vector<int> nodes;
			for(auto & iter : m_nodeInfo){
				nodes.push_back(iter.nodeId);
			}
			return nodes;
		};

		int NumberOfNodes(){
			return m_nodeInfo.size();
		}

		int NumberOfEdges(){
			return m_edgeInfo.size();
		}

		vector<pair<int, int> > Edges(){
			vector<pair<int,int> > edges;
			for(auto & info : m_edgeInfo){
				edges.emplace_back(info.fromNode, info.toNode);
			}
			return edges;
		};

		void UpdateProbability(int fromNode, int toNode, double probability=0.0){
			int edgeIndex = getEdgeIndex(fromNode,toNode);
			m_edgeInfo[edgeIndex].probability=probability;
		};

		void UpdateEstimate(int fromNode, int toNode, double estimate=0.0){
			int edgeIndex = getEdgeIndex(fromNode,toNode);
			m_edgeInfo[edgeIndex].estimate=estimate;
		};

		bool IsLive(int fromNode, int toNode){
			int edgeIndex = getEdgeIndex(fromNode,toNode);
			return m_edgeInfo[edgeIndex].isLive;
		};

		double Probability(int fromNode, int toNode){
			int edgeIndex = getEdgeIndex(fromNode,toNode);
			return m_edgeInfo[edgeIndex].probability;
		};

		double Estimate(int fromNode, int toNode){
			int edgeIndex = getEdgeIndex(fromNode,toNode);
			return m_edgeInfo[edgeIndex].estimate;
		};

		void AddEdge(int fromNode, int toNode, double probability=0.0, double estimate=0.0){
			int edgeId = m_edgeInfo.size();
			auto iter = m_nodeToIndex.find(fromNode);
			auto iterToNode = m_nodeToIndex.find(toNode);
			if(iter == m_nodeToIndex.end()){
				bool isNew = false;
				auto p = make_pair(iter, isNew);
				p = m_nodeToIndex.insert(make_pair(fromNode, m_nodeInfo.size()));
				iter = p.first;
				m_nodeInfo.emplace_back(fromNode);
			} else {
				int fromIndex = getNodeIndex(fromNode);
				auto edgeIter = m_nodeInfo[fromIndex].successors.find(toNode);
				if(edgeIter != m_nodeInfo[fromIndex].successors.end()){
					m_edgeInfo[edgeIter->second].probability=probability;
					return; //Overwritten
				} 
			}

			if(iterToNode == m_nodeToIndex.end()){
				bool isNew = false;
				auto p = make_pair(iterToNode, isNew);
				p = m_nodeToIndex.insert(make_pair(toNode, m_nodeInfo.size()));
				iterToNode = p.first;
				m_nodeInfo.emplace_back(toNode);
			}
				
			EdgeInfo edgeInfo(fromNode, toNode, probability, estimate);
			//cout << "*"<< edgeInfo.fromNode << " " << edgeInfo.toNode << endl;
			int edgeIndex = m_edgeInfo.size();
			m_edgeInfo.push_back(edgeInfo);
			m_nodeInfo[iter->second].successors.insert(make_pair(toNode,edgeIndex));
			m_nodeInfo[iterToNode->second].predecessors.insert(make_pair(fromNode,edgeIndex));
		};

		vector<int> Successors(int node){
			assert(HasNode(node));
			vector<int> successors;
			int index = getNodeIndex(node);
			for (auto & iter : m_nodeInfo[index].successors){
				successors.push_back(iter.first);
			}
			//ASSERT(HasNode(node);
			return successors;
		};


		vector<int> Predecessors(int node){
			assert(HasNode(node));
			vector<int> predecessors;
			int index = getNodeIndex(node);
			for (auto & iter : m_nodeInfo[index].predecessors){
				predecessors.push_back(iter.first);
			}
			//ASSERT(HasNode(node);
			return predecessors;
		};

		

		int Timestep(int node){
			assert(HasNode(node));
			int index = getNodeIndex(node);
			return m_nodeInfo[index].timestep;
		};

		int InfluencedBy(int node){
			assert(HasNode(node));
			int index = getNodeIndex(node);
			return m_nodeInfo[index].influencedBy;
		};

		bool IsInfluenced(int node){
			assert(HasNode(node));
			int index = getNodeIndex(node);
			return m_nodeInfo[index].isInfluenced;
		};

		void SetIsInfluenced(int node, int influencer, int timestep){
			int index = getNodeIndex(node);
			m_nodeInfo[index].isInfluenced = true;
			m_nodeInfo[index].influencedBy = influencer;
			m_nodeInfo[index].timestep = timestep;
			int edgeIndex = getEdgeIndex(influencer, node);
			m_edgeInfo[edgeIndex].isLive = true;

		};

		

		int getEdgeIndex(int fromNode, int toNode){
			assert(HasNode(fromNode));
			int fromIndex = getNodeIndex(fromNode);
			auto iter = m_nodeInfo[fromIndex].successors.find(toNode);
			assert(iter != m_nodeInfo[fromIndex].successors.end());
			return iter->second;
		};


		int getNodeIndex(int node){
			return m_nodeToIndex.find(node)->second;
		};



	private:
		vector<NodeInfo>				m_nodeInfo;
		vector<EdgeInfo>				m_edgeInfo;
		unordered_map<int, int>			m_nodeToIndex;


};

int main(){
	Xgraph xg;
	xg.AddEdge(1,2,0.2);
	xg.AddEdge(3,4,0.1);
	xg.AddEdge(30,40,0.1);
	xg.AddEdge(1,7,0.2);
	xg.AddEdge(1,2,0.1);
	xg.AddEdge(4,2,0.1);


	vector<int> nodes = xg.Nodes();
	for (auto n : nodes){
		cout << n <<" ";
	} 
	cout << endl;

	vector<pair<int, int> > edges;
	edges = xg.Edges();
	for (auto e : edges){
		cout << e.first <<"->" << e.second <<" ";
	} 
	cout << endl;

	double probability = xg.Probability(1,2);
	double estimate = xg.Estimate(1,2);

	vector<int> successors = xg.Successors(1);

	for (auto n : successors){
		cout << n <<" ";
	} 
	cout << endl;

	vector<int> predecessors = xg.Predecessors(2);

	for (auto n : predecessors){
		cout << n <<" ";
	} 
	cout << endl;

	cout << "P: " << probability << endl;
	cout << "E: " << estimate << endl;
	xg.UpdateProbability(1,5,0.1);
	//xg.Successors(23);

	return 0;
}
