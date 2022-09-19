""" 최종 파일 ㅇㅇ """
import dgl

g = dgl.load_graphs('subsampling/dgl_graph_full_heterogeneous.bin')
print(g)

#%%
sub1 = dgl.node_subgraph(g[0][0], {'paper':[x for x in range(100000)],
                                   'institution':[x for x in range(10000)],
                                    'author':[x for x in range(100000)]
                                   })
print(g)
print(sub1)

dgl.save_graphs('subsampling/dgl_graph_homogeneous_sub100000.bin', sub1)
#%%
graph = dgl.load_graphs('subsampling/dgl_graph_homogeneous_sub100000.bin')
print('before')
print(g[0][0])
print(f'\nafter')
print(graph[0][0])
