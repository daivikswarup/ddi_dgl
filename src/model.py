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
from graphsage import GraphSAGE
from utils import get_paths

class Basis_linear(nn.Module):
    def __init__(self, input_size, output_size, bases, rel_names):
        super(Basis_linear, self).__init__()
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
        super(RGCN_layer,self).__init__()
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


class diffpool_layer_hetero(nn.Module):
    # diffpool using rgcn instead of graphsage for a heterograph
    def __init__(self, input_size, output_size, etypes,num_clusters,basis=30):
        super(diffpool_layer_hetero, self).__init__()
        self.embed_layer = RGCN_layer(input_size, output_size, etypes,basis)
        self.num_clusters = num_clusters
        self.assignment_layer = RGCN_layer(input_size, num_clusters,
                                           etypes, basis, 'None')
        self.ddi_edges = [e for e in etypes if e != 'dpi' and e!='pdi' and e != 'ppi']
        self.dpi_edges = 'dpi'
        self.pdi_edges = 'pdi'
        self.ppi_edges = 'ppi'

    def forward(self, g, data):
        embeddings = self.embed_layer(g, data)
        assignment = self.assignment_layer(g, data)
        assignment = {e:v.view(-1,2,self.num_clusters//2) for e,v in \
                      assignment.items()}

        assignment['drug'] = \
                          torch.cat([F.softmax(assignment['drug'][:,0,:],dim=1),\
                                torch.zeros_like(assignment['drug'][:,0,:])],dim=1)
        assignment['protien'] = \
                          torch.cat([torch.zeros_like(assignment['protien'][:,1,:]),\
                              F.softmax(assignment['protien'][:,1,:],dim=1)],dim=1)

        ddi_adjs = {e: g.adjacency_matrix(etype=e).to_dense().cuda() for e in self.ddi_edges}
        # rest have only 1 type
        dpi_adj = g.adjacency_matrix(etype=self.dpi_edges).to_dense().cuda()
        pdi_adj = g.adjacency_matrix(etype=self.pdi_edges).to_dense().cuda()
        ppi_adj = g.adjacency_matrix(etype=self.ppi_edges).to_dense().cuda()
        
        s_d = assignment['drug']
        s_dt = torch.transpose(s_d, 0, 1)
        s_p = assignment['protien']
        s_pt = torch.transpose(s_p, 0, 1)

        ddi_2 = {e: torch.matmul(torch.matmul(s_dt, mat), s_d) for e, mat in \
                                  ddi_adjs.items()}
        ppi_2 = torch.matmul(torch.matmul(s_pt, ppi_adj), s_p)
        pdi_2 = torch.matmul(torch.matmul(s_dt, pdi_adj), s_p) 
        dpi_2 = torch.matmul(torch.matmul(s_pt, dpi_adj), s_d) 
        adj2 = ddi_2
        adj2['pdi'] = pdi_2
        adj2['dpi'] = dpi_2
        adj2['ppi'] = ppi_2

        # adj2 = torch.matmul(torch.matmul(s_dt, ddi_adj), s_d) +\
        #         torch.matmul(torch.matmul(s_pt, ppi_adj), s_p)+\
        #         torch.matmul(torch.matmul(s_dt, pdi_adj), s_p)
        embedding2 = torch.matmul(s_dt,embeddings['drug']) + \
                     torch.matmul(s_pt, embeddings['protien'])
        # adj2 = num_clusers x num_clusters
        # embedding2 = num_clusters x output_size
        return adj2, embedding2, s_d, s_p, embeddings


class RGCN_layer_vectorized(nn.Module):

    """Docstring for RGCN_layer. """

    def __init__(self, input_size, output_size, etypes,\
                 basis=30,nonlinearity='ReLU'):
        """TODO: to be defined. """
        super(RGCN_layer_vectorized,self).__init__()
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

    def forward(self, adj, data):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """
        rel_mats = [torch.matmul(a, self.basis_layer(data, e)) for e, a in \
                                 adj.items()]
        output = self.self_loop(data) + torch.sum(torch.stack(rel_mats,\
                                                dim=0), dim=0)
        
        return output



# Diffpool layer that takes full adjacency matrix and cluster embeddings
class diffpool_layer_hetero_vectorized(nn.Module):
    def __init__(self, input_size, input_clusters, output_size,\
                 output_clusters, etypes, basis=30):
        super(diffpool_layer_hetero_vectorized, self).__init__()
        self.embedding_layer = RGCN_layer_vectorized(input_size, output_size,
                                                     etypes, basis) 
        self.assignment_layer = RGCN_layer_vectorized(input_size, output_size,
                                                     etypes, basis) 
    
    def forward(self, adj, emb):
        z = self.embedding_layer(adj, emb)
        s = F.softmax(self.assignment_layer(adj,emb))
        s_t = torch.transpose(s, 0, 1)
        adj2 = {e:torch.matmul(torch.matmul(s_t, a), s) for \
                e,a in adj.items()}
        emb2 = torch.matmul(s_t, emb)
        return emb2, adj2, s



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


class DiffPoolEncoder(nn.Module):
    # runs multiple layers of diffpool 
    # returns node embeddings, cluster embeddings and assignment
    def __init__(self, g, sizes, num_clusters):
        super(DiffPoolEncoder, self).__init__()
        self.first_diffpool = \
                      diffpool_layer_hetero(sizes[0],sizes[1],g.etypes,num_clusters[0])
        self.next_diffpool_layers = nn.ModuleList(\
                    [diffpool_layer_hetero_vectorized(ins,inc,outs,outc,g.etypes) for ins,outs, inc, outc  in \
                    zip(sizes[1:-1], sizes[2:],num_clusters[:-1],num_clusters[1:])])

        self.ip = \
        nn.ParameterDict({ntype:nn.Parameter(torch.zeros((g.number_of_nodes(ntype),sizes[0])).cuda())\
                          for ntype in g.ntypes})
        for ntype in g.ntypes:
            nn.init.xavier_uniform_(self.ip[ntype])

    def forward(self, g):
        adj, emb, s_d, s_p, orig = self.first_diffpool(g, self.ip)
        smaps = []
        for l in self.next_diffpool_layers:
            emb, adj, s = l(adj, emb)
            smaps.append(s)
        # assign nodes to final cluster
        if len(smaps):
            prod = smaps[0]
            for s in smaps[1:]:
                prod = torch.matmul(prod, s)
            s_d = torch.matmul(s_d, prod)
            s_p = torch.matmul(s_p, prod)
        # print(s_d[:2,:])
        # print(s_p[:2,:])
        
        #
        return orig, adj, emb, s_d, s_p

        #  drug_embeddings = torch.matmul(s_d, emb)
        #  protien_embeddings = torch.matmul(s_p, emb)
        #  return {'drug':drug_embeddings, 'protien': protien_embeddings},adj,\
        #                           emb, s_d, s_p



class LinkPredictionDiffpool(nn.Module):

    """Docstring for LinkPrediction. """

    def __init__(self, g, sizes=[128, 128, 128], num_clusters=[128,64]):
        """TODO: to be defined. """
        super(LinkPredictionDiffpool, self).__init__()
        self.encoder = DiffPoolEncoder(g, sizes, num_clusters)
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
        node_embeddings, adj, emb, s_d, s_p = self.encoder(g)
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
        
class LinkPredictionDiffpoolPA(nn.Module):

    """Docstring for LinkPrediction. """

    def __init__(self, g, sizes=[128, 128,128], num_clusters=[128, 64],
                 num_paths=5):
        """TODO: to be defined. """
        super(LinkPredictionDiffpoolPA, self).__init__()
        self.encoder = DiffPoolEncoder(g, sizes, num_clusters)
        # self.output_layer = nn.Linear(2*sizes[-1], self.output_size)
        self.relation_matrices = nn.Parameter(torch.zeros([len(g.etypes),
                                                            sizes[-1],
                                                            sizes[-1]]))
        self.sizes = sizes
        nn.init.xavier_normal_(self.relation_matrices)
        self.lstm = nn.LSTM(sizes[-1],sizes[-1],1)
        self.attention = nn.MultiheadAttention(sizes[-1], 2)
        self.output_layer = nn.Linear(3*sizes[-1], len(g.etypes))

    def get_path(self, path, embeddings):
        vecs = [embeddings[node,:] for node in path]
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
        node_embeddings, adj, emb, s_d, s_p = self.encoder(g)
        # assuming relations are only between drugs
        drug_start = node_embeddings['drug'][nodepairs[:,0]]
                                            # batch x 1 x dim
        drug_end = node_embeddings['drug'][nodepairs[:,1]]
                                            # batch x dim x 1
        paths = get_paths(adj, nodepairs, s_d, 5)
        path_embeddings = [[self.path_embedding(path, node_embeddings) for path in datum] for \
                                   datum in paths]
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
            
