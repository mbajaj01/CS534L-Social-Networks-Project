import pandas as pd

votes = pd.read_csv('digg_votes_date.csv')
prune_filter = votes['voter_id'].value_counts()>20
pruned_nodes = list(set(prune_filter[prune_filter].index))
print 'Total nodes after pruning ' + str(len(pruned_nodes))

print len(votes)
votes_pruned = votes[votes.voter_id.isin(pruned_nodes)]
votes_pruned.to_csv('digg_votes_date_pruned.csv',index=False)
print len(votes_pruned)

friends = pd.read_csv('digg_friends.csv')
print len(friends)
pruned_friends = [a and b for a, b in zip(friends.user_id.isin(pruned_nodes).tolist(), friends.friend_id.isin(pruned_nodes).tolist())]
friends_pruned = friends[pruned_friends]
friends_pruned.to_csv('digg_friends_pruned.csv',index=False)
print len(friends_pruned)