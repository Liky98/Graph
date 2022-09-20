import dgl
import numpy as np

g = dgl.load_graphs('subsampling/dgl_graph_full_heterogeneous.bin')
sub1 = dgl.node_subgraph(g[0][0], {'paper':[x for x in range(100000)],
                                   'institution':[x for x in range(10000)],
                                    'author':[x for x in range(100000)]
                                   })
dgl.save_graphs('subsampling/dgl_graph_homogeneous_sub.bin', sub1)
#graph = dgl.load_graphs('subsampling/dgl_graph_homogeneous_sub100000.bin')


#%%
a = np.load("num_authors.npy")
b = np.load("paper_label.npy")
c = np.load("train_idx.npy")
print(a)
print(b.shape)
print(c.shape)