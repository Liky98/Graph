import dgl

g = dgl.load_graphs('subsampling/dgl_graph_full_heterogeneous.bin')
g = g.formats(['coo'])



subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=1)
g1 = subgraph.formats(['csc'])
dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k1.bin', g1)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=2)
g2 = subgraph.formats(['csc'])
dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k2.bin', g2)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=3)
g3 = subgraph.formats(['csc'])
dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k3.bin', g3)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=4)
g4 = subgraph.formats(['csc'])
dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k4.bin', g4)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=5)
g5 = subgraph.formats(['csc'])
dgl.save_graphs('subsampling/dgl_graph_homogeneous_csc_k5.bin', g5)
