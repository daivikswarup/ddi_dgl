import torch
import numpy as np
import pandas as pd
from load_graph import build_multigraph
import networkx as nx
from itertools import islice


class GraphDataset:

    """Docstring for GraphDataset. """

    def __init__(self, drugs, protiens, relations, ddi, ppi, dpi):
        """TODO: to be defined. """

        self.ddi = ddi
        self.ppi = ppi
        self.dpi = dpi
        self.nmap = {drug:i for i, drug in enumerate(drugs)}
        self.pmap = {prot:i for i, prot in enumerate(protiens)}
        self.rmap = {rel:i for i, rel in enumerate(relations)}
        self.g, self.nx_g = build_multigraph(self.nmap, self.pmap,self.rmap, self.ddi, self.ppi, self.dpi)
        # for path in nx.all_simple_paths(self.nx_g, (1, 'drug'),
        #                                        (2,'drug'), cutoff=3):
        #     print(path)

    def __len__(self):
        """TODO: Docstring for __len__.
        :returns: TODO

        """
        return len(self.ddi)

    def get_paths(self, edges, npaths):
        paths = []
        for n1, n2 in edges:
            try:
                shortest_paths = list(islice(nx.all_simple_paths(self.nx_g, (int(n1), 'drug'),\
                  (int(n2), 'drug'), cutoff=3),npaths))
            except nx.exception.NetworkXNoPath: # no path between nodes
                shortest_paths = []
            paths.append(shortest_paths)
        return paths

    def get_batches(self, batch_size=16, corruption=1, npaths=5):
        """TODO: Docstring for __getitem__.

        :idx: TODO
        :returns: TODO

        """
        all_nodes = list(self.nmap.values())
        # random permutation of edges
        perm = np.random.permutation(len(self.ddi)) 
        for i in range(0, len(self.ddi), batch_size):
            selected = perm[i:i+batch_size]
            v1 = self.ddi.iloc[selected]['STITCH 1'].values.tolist() 
            v2 = self.ddi.iloc[selected]['STITCH 2'].values.tolist() 
            rel = self.ddi.iloc[selected]['Polypharmacy Side Effect'].values.tolist()
            n1 = [self.nmap[x] for x in v1]
            n2 = [self.nmap[x] for x in v2]
            r = [self.rmap[x] for x in rel]
            # as in Max welling rgcn paper, corrupt edges
            neg_n1, neg_n2, neg_r = [],[], []
            for j in range(len(n1)):
                for k in range(corruption):
                    random_node = np.random.choice(all_nodes)
                    neg_r.append(r[j])
                    if np.random.random() < 0.5:
                        neg_n1.append(random_node)
                        neg_n2.append(n2[j])
                    else:
                        neg_n1.append(n1[j])
                        neg_n2.append(random_node)
            edges = np.array(list(zip(n1,n2)))
            labels = np.array(r)
            neg_edges = np.array(list(zip(neg_n1,neg_n2)))
            neg_labels = np.array(neg_r)
            all_edges = torch.Tensor(np.concatenate([edges, neg_edges],0)).long()
            all_rels = torch.Tensor(np.concatenate([labels, neg_labels],0)).long()
            y = torch.Tensor([1]*len(n1)+[0]*len(neg_n1)).long()
            paths = self.get_paths(all_edges, npaths)
            yield self.g, all_edges, all_rels, y, paths
                        

if __name__ == "__main__":
    data = GraphDataset('../data/bio-decagon-combo.csv',
                        '../data/bio-decagon-ppi.csv',
                        '../data/bio-decagon-targets-all.csv')
    for g, e, l, y in data.get_batches():
        print(e)
        print(l)

