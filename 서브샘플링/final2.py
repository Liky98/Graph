import dgl

g = dgl.load_graphs('subsampling/dgl_graph_full_heterogeneous.bin')
print(g)
#%%
sub1 = dgl.node_subgraph(g[0][0], {'paper':[x for x in range(100)]})
print(sub1)
#g1 = sub1.formats(['csc'])
dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc2.bin', sub1)

#%%
# sub_g = dgl.node_subgraph(g[0][0], {'paper': [1, 2]})
# print(sub_g)
#
#
# #%%
# sub_g = dgl.node_subgraph(g[0][0], {'paper': [1, 2]})
# g1 = sub_g.formats(['csc'])
#
# dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc.bin', g1)


#%%
# subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=1)
# g1 = subgraph.formats(['csc'])
# dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k1.bin', g1)
#
# subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=2)
# g2 = subgraph.formats(['csc'])
# dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k2.bin', g2)
#
# subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=3)
# g3 = subgraph.formats(['csc'])
# dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k3.bin', g3)
#
# subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=4)
# g4 = subgraph.formats(['csc'])
# dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k4.bin', g4)
#
# subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=5)
# g5 = subgraph.formats(['csc'])
# dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k5.bin', g5)
