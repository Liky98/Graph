import ssl

ssl._create_default_https_context = ssl._create_unverified_context
"""
노드 집합 확장을 k번 반복한 다음 노드 유도 하위 그래프를 생성하여 얻습니다.
"""

import ogb
from ogb.lsc import MAG240MDataset
import tqdm
import numpy as np
import torch
import dgl
import dgl.function as fn
import argparse
import os

parser = argparse.ArgumentParser()
#parser.add_argument('--rootdir', type=str, default='../Dataset/', help='Directory to download the OGB dataset.')
parser.add_argument('--rootdir', type=str, default='dataset/', help='Directory to download the OGB dataset.')
parser.add_argument('--author-output-path', type=str, default='dataset/author.npy', help='Path to store the author features.')
parser.add_argument('--paper-output-path', type=str, default='dataset/paper.npy', help='Path to store the author features.')
parser.add_argument('--inst-output-path', type=str, default='dataset/inst.npy',
                    help='Path to store the institution features.')
parser.add_argument('--graph-output-path', type=str, default='dataset/Graph.dgl',help='Path to store the graph.')
parser.add_argument('--graph-format', type=str, default='csc', help='Graph format (coo, csr or csc).')
parser.add_argument('--graph-as-homogeneous', action='store_true', default=True, help='Store the graph as DGL homogeneous graph.')
parser.add_argument('--full-output-path', type=str, default='dataset/full.npy',
                    help='Path to store features of all nodes.  Effective only when graph is homogeneous.')
args = parser.parse_args()

print('Building graph')
dataset = MAG240MDataset(root=args.rootdir)
ei_writes = dataset.edge_index('author', 'writes', 'paper')
ei_cites = dataset.edge_index('paper', 'paper')
ei_affiliated = dataset.edge_index('author', 'institution')

# We sort the nodes starting with the papers, then the authors, then the institutions.
author_offset = 0
inst_offset = author_offset + dataset.num_authors
paper_offset = inst_offset + dataset.num_institutions

g = dgl.heterograph({
    ('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
    ('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
    ('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
    ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
    ('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
})

"""
위에까지는 그래프 만드는 코드이고, 잘돌아감.
문제는 아래에 있는 Feature값 만드는건데 메모리 오류뜸.
"""


paper_feat = dataset.paper_feat
# paper_feat = np.memmap(args.paper_output_path, mode='w+', dtype='float16',
#                         shape=(dataset.num_papers, dataset.num_paper_features))

author_feat = np.memmap(args.author_output_path, mode='w+', dtype='float16',
                        shape=(dataset.num_authors, dataset.num_paper_features))
inst_feat = np.memmap(args.inst_output_path, mode='w+', dtype='float16',
                      shape=(dataset.num_institutions, dataset.num_paper_features))

# Iteratively process author features along the feature dimension.
BLOCK_COLS = 16
with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
    for start in tq:
        tq.set_postfix_str('Reading paper features...')
        g.nodes['paper'].data['x'] = torch.FloatTensor(paper_feat[:, start:start + BLOCK_COLS].astype('float32'))
        # Compute author features...

        tq.set_postfix_str('Computing author features...')
        g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='write-by')

        # Then institution features...
        tq.set_postfix_str('Computing institution features...')
        g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliate-with')

        tq.set_postfix_str('Writing author features...')
        author_feat[:, start:start + BLOCK_COLS] = g.nodes['author'].data['x'].numpy().astype('float16')

        tq.set_postfix_str('Writing institution features...')
        inst_feat[:, start:start + BLOCK_COLS] = g.nodes['institution'].data['x'].numpy().astype('float16')

        del g.nodes['paper'].data['x']
        del g.nodes['author'].data['x']
        del g.nodes['institution'].data['x']
author_feat.flush()
inst_feat.flush()


subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=1)
g1 = subgraph.formats(args.graph_format)
dgl.save_graphs('dataset/Graph_k_1.dgl', g1)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=2)
g2 = subgraph.formats(args.graph_format)
dgl.save_graphs('dataset/Graph_k_2.dgl', g2)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=3)
g3 = subgraph.formats(args.graph_format)
dgl.save_graphs('dataset/Graph_k_3.dgl', g3)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=4)
g4 = subgraph.formats(args.graph_format)
dgl.save_graphs('dataset/Graph_k_4.dgl', g4)

subgraph, inverse_indices = dgl.khop_out_subgraph(g, {'paper': 1}, k=5)
g5 = subgraph.formats(args.graph_format)
dgl.save_graphs('dataset/Graph_k_5.dgl', g5)
