import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import csv
from datetime import datetime 


votes = pd.read_csv('data/digg_votes.csv')
print votes.head()

## add date to votes
utc = []
date = []

timestamps = votes['timestamp'].tolist()
len(votes)

for test_time in timestamps:
    utc_time = datetime.utcfromtimestamp(test_time)
    utc.append(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))
    date.append(utc_time.date())
    #print 'a'

#print date
votes['utctime'] = pd.Series(utc)
votes['date'] = pd.Series(date)
votes.to_csv('data/digg_votes_date.csv')
print 'done'