#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import dgl
from tqdm import tqdm
import os

def ddi_graph(data, nodemap, etypes=['']):
    """TODO: Docstring for red_csv.

    :data: pd csv
    :returns: TODO

    """
    # data = pd.read_csv(path)
    # nodeids = list(set(list(data['STITCH 1']) + list(data['STITCH 2'])))
    # nodeids = sorted(nodeids)
    # nodemap = {node:id for id, node in enumerate(nodeids)}

    edges = defaultdict(list)
    print(len(data))
    for i, row in tqdm(data.iterrows(), desc='loading ddi'):
        edges['drug', row['Polypharmacy Side Effect'], 'drug']\
                    .append([nodemap[row['STITCH 1']], nodemap[row['STITCH 2']]])
        edges['drug', row['Polypharmacy Side Effect'], 'drug']\
                    .append([nodemap[row['STITCH 2']], nodemap[row['STITCH 1']]])
    return edges
    

def ppi_graph(data, p_map):
    """TODO: Docstring for read_ppi.

    :data: TODO
    :returns: TODO

    """
    # data = pd.read_csv(path)
    # p_ids = list(set(list(data['Gene 1']) + list(data['Gene 2'])))
    # p_ids = sorted(p_ids)
    # p_map = {p:i for i, p in enumerate(p_ids)}
    edge_list = []
    for i, row in data.iterrows():
        edge_list.append([p_map[row['Gene 1']], p_map[row['Gene 2']]])
        edge_list.append([p_map[row['Gene 2']], p_map[row['Gene 1']]])
    edges = {('protien', 'ppi', 'protien'): edge_list}
    return edges


def dpi_graph(data, n_map, p_map):
    """Create drug protien edges with only the nodes and protiens in the
    arguments

    :data: TODO
    :n_map: TODO
    :p_map: TODO
    :returns: TODO

    """
    # data = pd.read_csv(path)
    edge_list_fwd = []
    edge_list_bwd = []
    for i, row in data.iterrows():
        if row['STITCH'] not in n_map or row['Gene'] not in p_map:
            continue
        edge_list_fwd.append([n_map[row['STITCH']], p_map[row['Gene']]])
        edge_list_bwd.append([p_map[row['Gene']], n_map[row['STITCH']]])
        if i > 100:
            break
    edges = {('protien', 'pdi', 'drug'): edge_list_bwd, \
             ('drug', 'dpi', 'protien'): edge_list_fwd}
    return edges

def build_multigraph(nmap, pmap, rmap, ddi_df, ppi_df, dpi_df):
    """TODO: Docstring for build_multigraph.
    :returns: TODO

    """
    ddi_edges = ddi_graph(ddi_df, nmap)
    ppi_edges = ppi_graph(ppi_df, pmap)
    pdi_edges = dpi_graph(dpi_df, nmap, pmap)
    all_edges = ppi_edges
    all_edges.update(ddi_edges)
    all_edges.update(pdi_edges)
    g = dgl.heterograph(all_edges, num_nodes_dict={'drug':len(nmap), \
                                                   'protien': len(pmap)})
    return g
    


if __name__ == "__main__":
    ddi = pd.read_csv('../data/bio-decagon-combo.csv', nrows=100)
    ppi = pd.read_csv('../data/bio-decagon-ppi.csv')
    dpi = pd.read_csv('../data/bio-decagon-targets-all.csv')
    g,_,_,_ = build_multigraph(ddi, ppi, dpi)
    print(g)
    
