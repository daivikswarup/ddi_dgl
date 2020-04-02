from main import read_data
from argparser import parse_args
from data import GraphDataset
import networkx as nx
import pickle
import os
path = '/mnt/nfs/work1/mccallum/dswarupogguv/x.pkl'
args = parse_args()
if os.path.exists(path):
    with open(path, 'rb') as f:
        nx_g = pickle.load(f)
else:
    drugs, protiens, relations, ddi, ppi, dpi = read_data(args)
    dataset = GraphDataset(drugs, protiens, relations,ddi, ppi, dpi)
    
    with open(path, 'wb') as f:
        pickle.dump(dataset.nx_g, f)
        nx_g = dataset.nx_g
print('loaded')
paths = list(nx.all_simple_paths(nx_g, (2, 'drug'),(1,'drug'), cutoff=4))
print('here')
print(len(paths))
#for path in paths:
#    print(path)
with open('path.pkl', 'wb') as f:
    pickle.dump(paths, f)
