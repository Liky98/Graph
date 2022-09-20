"""

필요한 내용

1. dataset.paper_label
2. dataset.get_idx_split() ##such as ('train') or ('valid')
3. dataset.paper_feat
4. dataset.num_authors
5. dataset.num_classes
6. num_institutions


아래와 같이 호출해야함.

    g.nodes['paper'].data['feat'] = features
    g.nodes['author'].data['feat'] = author_emb
    g.nodes['institution'].data['feat'] = institution_emb
    g.nodes['field_of_study'].data['feat'] = topic_emb


###################

g = dgl.load_graphs('subsampling/dgl_graph_homogeneous_sub100000.bin')[0][0]

1 : dataset.paper_label[:10000]


3:  dataset = MAG240MDataset(root = "C:/Users/LeeKihoon/Graph/Dataset/")
    print(dataset.num_paper_features) # 768


4 : g.number_of_nodes('author') # dataset.num_authors
4 : g.num_nodes('author') # also same


5:  dataset = MAG240MDataset(root = "C:/Users/LeeKihoon/Graph/Dataset/")
    print(dataset.num_classes) # 153


6 : g.number_of_nodes('institution') # dataset.num_authors
6 : g.num_nodes('institution') # also same

"""

import dgl
from ogb.lsc import MAG240MDataset
import numpy as np

print(' GRAPH \n ')
g = dgl.load_graphs('subsampling/dgl_graph_homogeneous_sub100000.bin')[0][0]
print(g)


print(' DATASET \n ')
dataset = MAG240MDataset(root = "C:/Users/LeeKihoon/Graph/Dataset/")
path =  'C:/Users/LeeKihoon/Graph/Dataset/mag240m_kddcup2021/processed/paper/node_feat.npy'

x = np.memmap(path, dtype=np.float16, mode='r')#, shape=(121751666,768))#,shape=(:768))



print(dataset.num_paper_features)
print(x.shape)
print(x[:10]) #
print(dataset.num_classes) # 153

split_dict = dataset.get_idx_split()

print(dataset.num_papers)
print(dataset.num_authors)
print(dataset.num_institutions)
print(dataset.paper_label[:10]) # numpy array of shape (num_papers, ), storing target labels of papers.
np.save("paper_label.npy",dataset.paper_label[:100000])
np.save("num_authors.npy", 100000)
np.save("num_institutions.npy",10000)
np.save("num_classes.npy",153)


# alternatively, you can do the following.
train_idx = dataset.get_idx_split('train')
valid_idx = dataset.get_idx_split('valid')
testdev_idx = dataset.get_idx_split('test-dev')
testchallenge_idx = dataset.get_idx_split('test-challenge')

train_idx_list = []
valid_idx_list = []
testdev_idx_list = []
testchallenge_idx_list = []

for i in train_idx :
    if i < 100000 :
        train_idx_list.append(i)
for i in valid_idx :
    if i < 100000 :
        valid_idx_list.append(i)
for i in testdev_idx :
    if i < 100000 :
        testdev_idx_list.append(i)
for i in testchallenge_idx :
    if i < 100000 :
        testchallenge_idx_list.append(i)

np.save("train_idx.npy",train_idx_list)
np.save("valid_idx.npy",valid_idx_list)
np.save("testdev_idx.npy",testdev_idx_list)
np.save("testchallenge_idx.npy",testchallenge_idx_list)