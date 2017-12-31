#define HEAD_INFO
#include "sfmt/SFMT.h"
using namespace std;
typedef double (*pf)(int,int); 
class Graph
{
    public:
        int n, m, k;
        vector<int> inDeg;
        vector<vector<int>> gT;
        vector<vector<int>> g;

        vector<vector<vector<double>>> probT;
        vector<vector<vector<double>>> prob;
        vector<vector<vector<double>>> probEstimate;
        vector<vector<vector<double>>> probTEstimate;
        map<int, map<int,int>> edgeMapping;
        map<int, map<int,int>> edgeTMapping;
        vector<bool> visit;
        vector<int> visit_mark;
        enum InfluModel {IC, LT};
        InfluModel influModel;
        void setInfuModel(InfluModel p){
            influModel=p;
        }

        string folder;
        string graph_file;
        int numTopics;
        void readNM(){
            ifstream cin((folder+"attribute.txt").c_str());
            ASSERT(!cin == false);
            string s;
            while(cin >> s){
                if(s.substr(0,2)=="n="){
                    n=atoi(s.substr(2).c_str());
                    continue;
                }
                if(s.substr(0,2)=="m="){
                    m=atoi(s.substr(2).c_str());
                    continue;
                }
                ASSERT(false);
            }
            visit_mark=vector<int>(n);
            visit=vector<bool>(n);
            cin.close();
        }
        void add_edge(int a, int b, vector<double> p){
            edgeMapping[a][b] = g[a].size();
            edgeTMapping[b][a] = gT[b].size();
            probT[b].push_back(vector<double>());
            prob[a].push_back(vector<double>());
            probEstimate[a].push_back(vector<double>());
            probTEstimate[b].push_back(vector<double>());
            // cout << a << " " << b << " ";
            for(int i=0;i<(int)p.size();i++){
                double randomNum = (double)rand()/(RAND_MAX);
                probT[b][probT[b].size() - 1].push_back(p[i]);
                prob[a][prob[a].size() - 1].push_back(p[i]);
                // cout << p[i] << " ";
                probEstimate[a][probEstimate[a].size() - 1].push_back(randomNum/1000);
                probTEstimate[b][probTEstimate[b].size() - 1].push_back(randomNum/1000);
            }
            // cout << endl;
            gT[b].push_back(a);
            g[a].push_back(b);
            inDeg[b]++;

        }
        vector<bool> hasnode;
        void readGraph(){
            FILE * fin= fopen((graph_file).c_str(), "r");
            ASSERT(fin != false);
            int readCnt=0;
            for(int i=0; i<m; i++){
                readCnt ++;
                int a, b;
                int c=fscanf(fin, "%d%d", &a, &b);
                vector<double> p(numTopics, 0.0);
                for(int j=0; j<numTopics; j++){
                    double q = 0.0;
                    c = fscanf(fin, "%lf", &q);
                    p[j] = q;
                }
                // ASSERT(c==3);
                // ASSERTT(c==3, a, b, p, c);
                ASSERT( a<n );
                ASSERT( b<n );
                hasnode[a]=true;
                hasnode[b]=true;
                add_edge(a, b, p);
            }
            if(readCnt !=m)
                ExitMessage("m not equal to the number of edges in file "+graph_file);
            fclose(fin);
        }

        Graph(string folder, string graph_file, int topics):folder(folder), graph_file(graph_file){
            readNM();
            numTopics = topics;
            //init vector
            FOR(i, n){
                gT.push_back(vector<int>());
                g.push_back(vector<int>());
                hasnode.push_back(false);
                probT.push_back(vector<vector<double>>());
                prob.push_back(vector<vector<double>>());
                probEstimate.push_back(vector<vector<double>>());
                probTEstimate.push_back(vector<vector<double>>());
                edgeMapping[i] = map<int,int>();
                edgeTMapping[i] = map<int,int>();
                //hyperGT.push_back(vector<int>());
                inDeg.push_back(0);
            }

            readGraph();
        }

};
double sqr(double t)
{
    return t*t;
}

#include "infgraph.h"
#include "timgraph.h"


