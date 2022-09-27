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