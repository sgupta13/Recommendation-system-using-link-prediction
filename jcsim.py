
# coding: utf-8

# In[17]:

import numpy as np
import networkx as nx
import os
import copy
import sys
import itertools
from collections import defaultdict


# In[7]:

os.getcwd()


# In[8]:

def get_graph():
    os.chdir('D:\Graph Data Mining Final Project')
    data = np.loadtxt('flickrsauop.txt', usecols=(0,1))
    return nx.from_edgelist(data)


# In[9]:

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


# In[56]:

jc = list(nx.jaccard_coefficient(graph))


# In[111]:

def get_pred(node):
    for a in jc:
        i, j , l = a
        if (i == node) and l > 0.1:
            print(sorted(a, reverse=True))


# In[ ]:

for node in range(10000):
    get_pred(node)


# In[21]:

if type(graph) == nx.MultiGraph or type(graph) == nx.MultiDiGraph:
    print("simrank() not defined for graphs with multiedges.")


# In[23]:

if graph.is_directed():
    print("simrank() not defined for directed graphs.")


# In[18]:

def simrank(G, r=0.8, max_iter=100, eps=1e-4):
    nodes = G.nodes()
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}
    sim_prev = np.zeros(len(nodes))
    sim = np.identity(len(nodes))
    for i in range(max_iter):
        if np.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = np.copy(sim)
        for u, v in itertools.product(nodes, nodes):
            if u is v:
                continue
            u_ns, v_ns = G.predecessors(u), G.predecessors(v)
            # evaluating the similarity of current iteration nodes pair
            if len(u_ns) == 0 or len(v_ns) == 0: 
                # if a node has no predecessors then setting similarity to zero
                sim[nodes_i[u]][nodes_i[v]] = 0
            else:                    
                s_uv = sum([sim_prev[nodes_i[u_n]][nodes_i[v_n]] for u_n, v_n in itertools.product(u_ns, v_ns)])
                sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ns) * len(v_ns))
    return sim


# In[19]:

print(simrank(graph).round(3))


# In[36]:

sim_old = defaultdict(list)
sim = defaultdict(list)
for n in graph.nodes():
    sim[n] = defaultdict(int)
    sim[n][n] = 1
    sim_old[n] = defaultdict(int)
    sim_old[n][n] = 0


# In[37]:

def _is_converge(s1, s2, eps=1e-4):
    for i in s1.keys():
        for j in s1[i].keys():
            if abs(s1[i][j] - s2[i][j]) >= eps:
                return False
    return True


# In[ ]:

# calculate simrank
for iter_ctr in range(100):
    if _is_converge(sim, sim_old):
        break
    sim_old = copy.deepcopy(sim)
    for u in graph.nodes():
        for v in graph.nodes():
            if u == v:
                continue
            s_uv = 0.0
            for n_u in graph.neighbors(u):
                for n_v in graph.neighbors(v):
                    s_uv += sim_old[n_u][n_v]
            sim[u][v] = (0.85 * s_uv / (len(graph.neighbors(u)) * len(graph.neighbors(v))))
                
print(sim)


# In[ ]:



