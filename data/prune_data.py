import pandas as pd

votes = pd.read_csv('digg_votes_date.csv')
prune_filter = votes['voter_id'].value_counts()>20
pruned_nodes = list(set(prune_filter[prune_filter].index))

# prune friends
friends = pd.read_csv('digg_friends.csv')
print len(friends)
pruned_friends = [a and b for a, b in zip(friends.user_id.isin(pruned_nodes).tolist(), friends.friend_id.isin(pruned_nodes).tolist())]
friends_pruned = friends[pruned_friends]
friends_pruned.to_csv('digg_friends_pruned.csv',index=False)
print len(friends_pruned)


# remove nodes which are there in votes but not in friends
print len(votes)
votes_pruned = votes[votes.voter_id.isin(pruned_nodes)]
print len(votes_pruned)
nodes_friends = list(sorted(set(list(sorted(set(friends_pruned.friend_id.tolist()))) + list(sorted(set(friends_pruned.user_id.tolist()))))))
votes_pruned = votes_pruned[votes_pruned.voter_id.isin(nodes_friends)]
print len(votes_pruned)

stories = list(set(votes_pruned['story_id']))
total_count = 0
for story in stories:
    votes_sub = votes_pruned[votes_pruned['story_id'] == story]
    
    hist = votes_sub.voter_id.value_counts()
    count_list = hist[hist>1].tolist()
    index_list = hist[hist>1].index.tolist()
    print count_list
    total_count = total_count + sum(count_list)
print total_count


# remove all adoptions after first adoption
drop_ind = []
stories = list(set(votes_pruned['story_id']))
for story in stories:
    votes_sub = votes_pruned[votes_pruned['story_id'] == story]    
    hist = votes_sub.voter_id.value_counts()
    count_list = hist[hist>1].tolist()
    index_list = hist[hist>1].index.tolist()
    for item in index_list:
        simar = votes_sub[votes_sub.voter_id == item]
        drop_list= simar.index.tolist()
        drop_ind = drop_ind + drop_list[1:]
        #drop_ind.append(drop_list[1:])
        
votes_pruned.drop(drop_ind, inplace=True)    

stories = list(set(votes_pruned['story_id']))
total_count = 0
for story in stories:
    votes_sub = votes_pruned[votes_pruned['story_id'] == story]
    
    hist = votes_sub.voter_id.value_counts()
    count_list = hist[hist>1].tolist()
    index_list = hist[hist>1].index.tolist()
    total_count = total_count + sum(count_list)
print total_count

print len(votes_pruned)
votes_pruned.to_csv('digg_votes_date_pruned.csv',index=False)
