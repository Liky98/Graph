import torch
from torch import nn
from tqdm import tqdm,trange
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from ogb.lsc import MAG240MDataset
import numpy as np
import ssl

#%%
ssl._create_default_https_context = ssl._create_unverified_context
dataset = DglNodePropPredDataset(name='ogbn-mag')
graph,label = dataset[0]
print(graph)
print(graph.num_nodes('author'))
print(len(graph.nodes['paper'].data['feat']))
print(graph.nodes)
#%%
import dgl
dgl.save_graphs('원본.bin', graph)
#%%
print(graph)
#%%
print(graph)
print(1)
paper_feat = graph.nodes['paper'].data['feat']
print(2)
author_feat = np.memmap("./author_output_path", mode='w+', dtype='float16', shape=(graph.num_nodes('author'), len(graph.nodes['paper'].data['feat'])))
print(3)
inst_feat = np.memmap("./inst_output_path", mode='w+', dtype='float16', shape=(graph.num_nodes('institution'), len(graph.nodes['paper'].data['feat'])))

#%%
BLOCK_COLS = 16
with trange(0, len(graph.nodes['paper'].data['feat']), BLOCK_COLS) as tq:
    for start in tq:
        tq.set_postfix_str('Reading paper features...')
        graph.nodes['paper'].data['x'] = torch.FloatTensor(paper_feat[:, start:start + BLOCK_COLS])
        # Compute author features...
        tq.set_postfix_str('Computing author features...')
        graph.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='writes')
        # Then institution features...
        tq.set_postfix_str('Computing institution features...')
        graph.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliated_with')
        tq.set_postfix_str('Writing author features...')
        author_feat[:, start:start + BLOCK_COLS] = graph.nodes['author'].data['x'].numpy().astype('float16')
        tq.set_postfix_str('Writing institution features...')
        inst_feat[:, start:start + BLOCK_COLS] = graph.nodes['institution'].data['x'].numpy().astype('float16')
        del graph.nodes['paper'].data['x']
        del graph.nodes['author'].data['x']
        del graph.nodes['institution'].data['x']
author_feat.flush()
inst_feat.flush()

print(graph)
# The FEATURE needs CSR graph
g = graph.formats(['csr'])
print(g)
dgl.save_graphs('특징추가그래프.bin', g)

#%%


g = dgl.load_graphs('원본.bin')
print(g[0][0].num_nodes('author'))
#%%
# #%%
# import dgl
#
#
# g, init_labels = dataset[0]
#
# #%%
#
# # Iteratively process author features along the feature dimension.
# BLOCK_COLS = 16
#
#
# with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
#     for start in tq:
#         tq.set_postfix_str('Reading paper features...')
#         g.nodes['paper'].data['x'] = torch.FloatTensor(paper_feat[:, start:start + BLOCK_COLS].astype('float32'))
#         # Compute author features...
#         tq.set_postfix_str('Computing author features...')
#         g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='writed_by')
#         # Then institution features...
#         tq.set_postfix_str('Computing institution features...')
#         g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliated_with')
#         tq.set_postfix_str('Writing author features...')
#         author_feat[:, start:start + BLOCK_COLS] = g.nodes['author'].data['x'].numpy().astype('float16')
#         tq.set_postfix_str('Writing institution features...')
#         inst_feat[:, start:start + BLOCK_COLS] = g.nodes['institution'].data['x'].numpy().astype('float16')
#         del g.nodes['paper'].data['x']
#         del g.nodes['author'].data['x']
#         del g.nodes['institution'].data['x']
# author_feat.flush()
# inst_feat.flush()
#
# # The FEATURE needs CSR graph
# g = g.formats(['csr'])
# dgl.save_graphs('test1.bin', g)