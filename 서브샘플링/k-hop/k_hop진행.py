"""
1-hop: 25 nodes
2-hop: 15 nodes
3-hop: 10 nodes
"""
import dgl
import numpy as np

g = dgl.load_graphs('../subsampling/dgl_graph_full_heterogeneous.bin')

g1 = g[0]
g2 = g[0][0]
print(g2)
#%%
#sg, inverse_indices = dgl.khop_in_subgraph(g2, {'paper':[0]}, k=2)
#print(sg)
#print(inverse_indices)

g= g2
sg = dgl.sampling.sample_neighbors(g= g, nodes =  {'paper':[0]}, fanout=1)
print(sg)
