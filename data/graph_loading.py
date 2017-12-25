import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

csv_path = "digg_friends.csv"
csv_path = "digg_friends_pruned.csv"
csv_path = "digg_simar.csv"

def load_graph(csv_path):
    G = nx.DiGraph()
    with open(csv_path, "rb") as f_obj:
        reader = csv.reader(f_obj)
        flag=1
        for row in reader:
            if flag==1:
                flag=0
                continue
            mutual = int(row[0].strip())
            timestamp = row[1]
            user_id = int(row[2])
            friend_id = int(row[3])
            prob1 = random.random()
            prob2 = random.random()
            if mutual==0:
                G.add_edge(user_id, friend_id,prob=prob1)
            elif mutual==1:
                G.add_edge(user_id, friend_id,prob = prob1)
                G.add_edge(friend_id, user_id,prob = prob2)
            else:
                print mutual
                print user_id
                print friend_id
                print 'mutual value range out of bound'
    return G

        

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

