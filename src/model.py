#!/usr/bin/env python

import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import numpy as np

class RGCN_layer(nn.Module):

    """Docstring for RGCN_layer. """

    def __init__(self, input_size, output_size, etypes):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        self.layer_dict = nn.ModuleDict({etype: nn.Linear(input_size, output_size)
                                    for etype in etypes})
        self.self_loop = nn.Linear(input_size, output_size)

    def forward(self, g, data):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """
        message_func_dict = {}
        for src, etype, dst in g.canonical_etypes:
            g.nodes[src].data['W_%s'%etype] = self.layer_dict[etype](data[src])
            message_func_dict[etype] = (fn.copy_u('W_%s'%etype, 'm'), \
                                        fn.mean('m', 'h'))
        for ntype in g.ntypes:
            g.nodes[ntype].data['Self_message'] = self.self_loop(data[ntype])
        
        g.multi_update_all(message_func_dict, 'sum')
        return {ntype:g.nodes[ntype].data['Self_message'] + g.nodes[ntype].data['h'] for\
                    ntype in g.ntypes}

# Mostly based on https://docs.dgl.ai/en/0.4.x/tutorials/hetero/1_basics.html
class RGCN(nn.Module):

    """2 RGCN layers"""

    def __init__(self, g, sizes=[128,128,128]):
        """TODO: to be defined.

        :g: TODO
        :sizes: TODO

        """
        nn.Module.__init__(self)
        # self.embeddings = nn.ParameterDict({ntype: \
        #             nn.Parameter(torch.Tensor(g.number_of_nodes(ntype),sizes[0]))\
        #                                     for ntype in g.ntypes})
        # for _, param in self.embeddings.items():
        #     nn.init.xavier_uniform_(param)
        self.num_nodes = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
        self.total = np.sum(list(self.num_nodes.values()))
        sizes = [self.total] + sizes 
        self.layers = nn.ModuleList([RGCN_layer(a, b, g.etypes) for a, b in \
                                     zip(sizes[:-1], sizes[1:])])
        self.ip = {}
        cum_sum = 0
        eye = np.eye(self.total)
        for ntype in sorted(g.ntypes):
            self.ip[ntype] = torch.Tensor(eye[cum_sum+np.arange(g.number_of_nodes(ntype))]).long()
            cum_sum += g.number_of_nodes(ntype)

        # self.ip = {ntype: torch.eye(g.number_of_nodes(ntype)) for ntype in g.ntypes}
    def forward(self, g):
        """TODO: Docstring for forward.

        :g: TODO
        :returns: TODO

        """
        op = self.ip
        for layer in self.layers:
            op = layer(g, op)
        return op


class LinkPrediction(nn.Module):

    """Docstring for LinkPrediction. """

    def __init__(self, g, sizes=[128,128,128]):
        """TODO: to be defined. """
        super(LinkPrediction, self).__init__()
        self.rgcn = RGCN(g, sizes)
        # self.output_layer = nn.Linear(2*sizes[-1], self.output_size)
        self.relation_matrices = nn.Parameter(torch.zeros([len(g.etypes),
                                                            sizes[-1],
                                                            sizes[-1]]))
        nn.init.xavier_normal_(self.relation_matrices)

    def forward(self, g, nodepairs, relations=None):
        """Pass graph through rgcn, predict edge labels for given edge pairs

        :g: TODO
        :edgepairs: TODO
        :returns: TODO

        """
        node_embeddings = self.rgcn(g)
        # assuming relations are only between drugs
        drug_start = node_embeddings['drug'][nodepairs[:,0]].unsqueeze(1)
                                            # batch x 1 x dim
        drug_end = node_embeddings['drug'][nodepairs[:,1]].unsqueeze(2)
                                            # batch x dim x 1
        if relations is not None:
            matrices = self.relation_matrices[relations] # batch x dim x dim
            return torch.matmul(torch.matmul(drug_start, matrices), \
                            drug_end).squeeze(1).squeeze(1)
        else:
            matrices = self.relation_matrices.unsqueeze(0)
            drug_start = drug_start.unsqueeze(1)
            drug_end = drug_end.unsqueeze(1)
            # 1 x classes x dim x dim
            # drug_start = b x 1 x 1 x dim
            # drug_end = b x 1 x dim x 1
            # output = b x classes
            logits = torch.matmul(torch.matmul(drug_start, matrices), \
                            drug_end).squeeze(-1).squeeze(-1)
        

        


def main():
    import load_graph
    g = load_graph.build_multigraph()
    gcn_model = RGCN(g)
    output = gcn_model(g)
    print(output)

    

if __name__ == "__main__":
    main()
            
