
# coding: utf-8

# In[7]:

import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt


# In[8]:

os.getcwd()


# In[9]:

def get_graph():
    os.chdir('D:\Graph Data Mining Final Project')
    data = np.loadtxt('flickr31.txt', usecols=(0,1))
    return nx.from_edgelist(data)


# In[ ]:

graph = get_graph()


# In[ ]:

jc = list(nx.jaccard_coefficient(graph))


# In[ ]:

for i,j,l in jc:
    if l > 0 :
        print(i, j, l)

