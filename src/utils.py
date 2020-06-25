import torch
from itertools import islice
import torch.nn as nn
from torch.nn import functional as F
import networkx as nx
import numpy as np



def get_paths(adj, nodepairs, s, npaths):
    adj = torch.sum(torch.stack([v for e,v in adj.items()],dim=0),dim=0)
    assignment = torch.argmax(s, dim=1)
    clusterpairs = [(assignment[a],assignment[b]) for a,b in nodepairs]
    npadj = adj.detach().cpu().numpy()
    threshold = np.percentile(npadj,90)
    npadj = npadj > threshold
    g = nx.from_numpy_matrix(npadj)
    paths = []
    for src, target in clusterpairs:
        try:
            src = int(src.detach().cpu().numpy())
            target = int(target.detach().cpu().numpy())
            shortest_paths = list(islice(nx.all_simple_paths(g, src, \
                    target,cutoff=3), npaths))
        except nx.exception.NetworkXNoPath:
            shortest_paths = []
        paths.append(shortest_paths)
    return paths


