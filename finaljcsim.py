
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


# In[ ]:

def simrank(G, r=0.8, max_iter=100, eps=1e-4):

    nodes = G.nodes()
    nodes_i = {k: v for(k, v) in [(nodes[i], i) for i in range(0, len(nodes))]}

    sim_prev = numpy.zeros(len(nodes))
    sim = numpy.identity(len(nodes))

    for i in range(max_iter):
        if numpy.allclose(sim, sim_prev, atol=eps):
            break
        sim_prev = numpy.copy(sim)
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


# In[ ]:

pprint(simrank(graph).round(3))

