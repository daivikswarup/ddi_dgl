import torch
import torch.nn as nn
from torch.nn import functional as F
## Taken from https://github.com/dmlc/dgl/blob/master/examples/pytorch/diffpool/model/tensorized_layers/graphsage.py

class GraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight,
            gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):

        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.relu(h_k)
        return h_k

    def __repr__(self):
            return super(GraphSAGE, self).__repr__()
