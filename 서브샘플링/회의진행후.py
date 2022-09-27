"""
1-hop: 25 nodes
2-hop: 15 nodes
3-hop: 10 nodes
"""
import dgl
import numpy as np

g = dgl.load_graphs('subsampling/dgl_graph_full_heterogeneous.bin')
#%%
g1 = g[0]
g2 = g[0][0]
print(g2)
print(g2.nodes['author'].data['x'])

#%%
sub1 = dgl.node_subgraph(g[0][0], {'paper':[x for x in range(100000)],
                                   'institution':[x for x in range(10000)],
                                    'author':[x for x in range(100000)]
                                   })
dgl.save_graphs('subsampling/dgl_graph_homogeneous_sub.bin', sub1)
#graph = dgl.load_graphs('subsampling/dgl_graph_homogeneous_sub100000.bin')

a = np.load("num_authors.npy")
b = np.load("paper_label.npy")
c = np.load("train_idx.npy")
print(a)
print(b.shape)
print(c.shape)

#%%

sg, inverse_indices = dgl.khop_in_subgraph(g, {'paper':[0,1,2]},k=2)
print(sg)
print(inverse_indices)


#%%
import dgl
from ogb.lsc import MAG240MDataset
path="-------------"
dataset = MAG240MDataset(root=path)
ei_cites = dataset.edge_index('paper', 'paper')

author_offset = 0
inst_offset = author_offset + dataset.num_authors
paper_offset = inst_offset + dataset.num_institutions


g = dgl.heterograph({
    ('paper', 'cites', 'paper'): (ei_cites[0], ei_cites[1]),
    ('paper', 'cited_by', 'paper'): (ei_cites[1], ei_cites[0])
})

dgl.save_graphs('subsampling/paper_paper.bin', g)