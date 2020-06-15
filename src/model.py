#!/usr/bin/env python

import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import numpy as np
from dgl.nn.pytorch.conv import RelGraphConv

class Basis_linear(nn.Module):
    def __init__(self, input_size, output_size, bases, rel_names):
        nn.Module.__init__(self)
        num_relations = len(rel_names)
        self.rel_names = rel_names
        self.bases = bases
        self.w = nn.Parameter(torch.zeros((bases,input_size,output_size)).cuda())
        self.coeff_mat = nn.Parameter(torch.zeros((num_relations,bases)).cuda())
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.coeff_mat)
        self.coeff = {e:w for e, w in zip(rel_names,\
                                          torch.unbind(self.coeff_mat,dim=0))}

    def forward(self, inp, rel_type):
        weight = torch.sum(self.coeff[rel_type].view(self.bases,1,1)*self.w,dim=0)
        return torch.matmul(inp,weight)
        


class RGCN_layer(nn.Module):

    """Docstring for RGCN_layer. """

    def __init__(self, input_size, output_size, etypes,\
                 basis=30,nonlinearity='ReLU'):
        """TODO: to be defined. """
        nn.Module.__init__(self)
        self.output_size = output_size
        self.basis_layer = Basis_linear(input_size, output_size, basis, etypes)
        self.self_loop = nn.Linear(input_size, output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.zeros_(self.bias)
        if nonlinearity == 'ReLU':
            self.nonlinearity = nn.ReLU()
        else:
            # No non linearity
            self.nonlinearity = nn.Sequential()

    def forward(self, g, data):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """
        for ntype in g.ntypes:
            g.nodes[ntype].data['h'] = torch.zeros((g.number_of_nodes(ntype),\
                                                    self.output_size)).cuda()
        message_func_dict = {}
        for src, etype, dst in g.canonical_etypes:
            g.nodes[src].data['W_%s'%etype] = self.basis_layer(data[src],etype)
            message_func_dict[etype] = (fn.copy_u('W_%s'%etype, 'm'), \
                                        fn.mean('m', 'h'))
        g.multi_update_all(message_func_dict, 'sum')
        for ntype in g.ntypes:
            g.nodes[ntype].data['Self_message'] = self.self_loop(data[ntype])
        
        return {ntype:self.nonlinearity(g.nodes[ntype].data['Self_message'] + \
                g.nodes[ntype].data['h'] +self.bias) for\
                    ntype in g.ntypes}

# Mostly based on https://docs.dgl.ai/en/0.4.x/tutorials/hetero/1_basics.html
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py
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
        # self.total = np.sum(list(self.num_nodes.values()))
        # sizes = [self.total] + sizes 
        self.layers = nn.ModuleList([RGCN_layer(a, b, g.etypes) for a, b in \
                                  zip(sizes[:-1], sizes[1:])])

        self.ip = \
        nn.ParameterDict({ntype:nn.Parameter(torch.zeros((g.number_of_nodes(ntype),sizes[0])).cuda())\
                          for ntype in g.ntypes})
        for ntype in g.ntypes:
            nn.init.xavier_uniform_(self.ip[ntype])
            


        # self.layers = nn.ModuleList([RelGraphConv(a, b, len(g.etypes),\
        #                                           num_bases=30) for a, b in \
        #                              zip(sizes[:-1], sizes[1:])])
        # self.ip = {}
        # cum_sum = 0
        # eye = np.eye(self.total)
        # for ntype in sorted(g.ntypes):
        #     self.ip[ntype] = \
        #            torch.Tensor(eye[cum_sum+np.arange(g.number_of_nodes(ntype))]).cuda()
        #     cum_sum += g.number_of_nodes(ntype)

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

    def __init__(self, g, sizes=[128, 128,128]):
        """TODO: to be defined. """
        super(LinkPrediction, self).__init__()
        self.rgcn = RGCN(g, sizes)
        # self.output_layer = nn.Linear(2*sizes[-1], self.output_size)
        self.relation_matrices = nn.Parameter(torch.zeros([len(g.etypes),
                                                            sizes[-1],
                                                            sizes[-1]]))
        nn.init.xavier_normal_(self.relation_matrices)

    def forward(self, g, nodepairs, relations=None, paths=None):
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

class PathAttention(nn.Module):

    """Docstring for LinkPrediction. """

    def __init__(self, g, sizes=[128, 128,128]):
        """TODO: to be defined. """
        super(PathAttention, self).__init__()
        self.sizes = sizes
        self.rgcn = RGCN(g, sizes)
        self.lstm = nn.LSTM(sizes[-1],sizes[-1],1)
        self.attention = nn.MultiheadAttention(sizes[-1], 2)
        self.output_layer = nn.Linear(3*sizes[-1], len(g.etypes))

    def get_path(self, path, embeddings):
        vecs = []
        for node, ntype in path:
            vecs.append(embeddings[ntype].index_select(0,torch.tensor(node).cuda()))
        # returns pathlen x dim
        return torch.stack(vecs, 0)

    def path_embedding(self, path, embeddings):
        inp = self.get_path(path, embeddings)
        # inp = inp.unsqueeze(1)
        op, (h, c) = self.lstm(inp)
        return h.view(-1)

    def get_attention_context(self, path_embeddings, start, end):
        if len(path_embeddings) == 0: #no paths
            return torch.zeros(self.sizes[-1]).cuda()
        stacked = torch.stack(path_embeddings, 0).unsqueeze(1) # NPath x 1 x dim
        start = start.unsqueeze(0).unsqueeze(0) # 1 x 1 x dim
        attn, att_wts = self.attention(start, stacked, stacked)
        return attn.squeeze(0).squeeze(0)

    def forward(self, g, nodepairs, relations=None, paths=None):
        """Pass graph through rgcn, predict edge labels for given edge pairs

        :g: TODO
        :edgepairs: TODO
        :returns: TODO

        """
        node_embeddings = self.rgcn(g)
        path_embeddings = [[self.path_embedding(path, node_embeddings) for path in datum] for \
                                   datum in paths]


        # assuming relations are only between drugs
        drug_start = node_embeddings['drug'][nodepairs[:,0]]
                                            # batch x 1 x dim
        drug_end = node_embeddings['drug'][nodepairs[:,1]]
                                            # batch x dim x 1
        context = [self.get_attention_context(pe, start, end) for\
                   pe, start, end in zip(path_embeddings, torch.unbind(drug_start), \
                           torch.unbind(drug_end))]
        context = torch.stack(context) # Batch x dim
        all_features = torch.cat([drug_start, drug_end, context],-1)
        op = self.output_layer(all_features)
        if relations is not None:
            return torch.gather(op, 1, relations.unsqueeze(-1)).squeeze(-1)
        else:
            return op
        

def main():
    import load_graph
    g = load_graph.build_multigraph()
    gcn_model = RGCN(g)
    output = gcn_model(g)
    print(output)

    

if __name__ == "__main__":
    main()
            
