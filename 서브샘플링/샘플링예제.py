import dgl
import torch

g = dgl.heterograph({
    ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1]),
    ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2])
})

print(g)

sub_g = dgl.node_subgraph(g, {'user': [1, 2]})
print(sub_g)