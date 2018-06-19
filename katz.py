
# coding: utf-8

# In[1]:

import numpy as np
import networkx as nx
import os


# In[2]:

os.getcwd()


# In[3]:

def get_graph():
    os.chdir('D:\Graph Data Mining Final Project')
    data = np.loadtxt('flickrsauop.txt', usecols=(0,1))
    return nx.from_edgelist(data)


# In[4]:

graph = get_graph()


# In[5]:

graph_adj_matrix = nx.adjacency_matrix(graph)


# In[6]:

graph_adj_matrix = graph_adj_matrix.todense()


# In[11]:

c = 0.85
w, v = np.linalg.eigh(graph_adj_matrix)
lambda1 = max([abs(x) for x in w])
I = np.eye(graph_adj_matrix.shape[0])
S = None
S = np.linalg.inv(I - c/lambda1 * graph_adj_matrix)


# In[52]:

def get_pred(node):
        arr = np.array(S[node])[0]
        ix = arr.argsort()[-5:]
        return dict(zip(ix, arr[ix]))


# In[55]:

for node in range(10000):
    print(get_pred(node))


# In[ ]:

jc = list(nx.jaccard_coefficient(graph))

