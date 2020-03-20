#!/usr/bin/env python

from argparser import parse_args
from model import LinkPrediction 
from tqdm import tqdm
from data import GraphDataset
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold


def filter_ddi(ddi, mincount=20000):
        # remove infrequent edges
        counts = ddi['Polypharmacy Side Effect'].value_counts()
        counts = counts[counts >= mincount]
        df_edge_types = counts.index.values.tolist()
        df_index = ddi[ddi['Polypharmacy Side Effect'].isin(df_edge_types)].index.values.tolist()  #[:50]  # for testing locally
        ddi = ddi.iloc[df_index]
        return ddi

def read_data(args):
    """TODO: Docstring for read_data.

    :args: TODO
    :returns: TODO

    """
    ddi = filter_ddi(pd.read_csv(args.ddi))
    ppi = pd.read_csv(args.ppi)
    dpi = pd.read_csv(args.dpi)
    drugs = sorted(list(set(ddi['STITCH 1'].values.tolist() \
                     + ddi['STITCH 2'].values.tolist())))
    protiens = sorted(list(set(ppi['Gene 1'].values.tolist()\
                        + ppi['Gene 2'].values.tolist())))
    relations = sorted(list(set(ddi['Polypharmacy Side Effect'])))
    return drugs, protiens,relations, ddi, ppi, dpi

def train(model, dataset, args):
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(args.num_epochs):
        for i, (g, e, l, y) in tqdm(enumerate(dataset.get_batches(args.batch_size)),
                        desc='Epoch %d'%epoch,total=len(dataset)/args.batch_size):
            optimizer.zero_grad()
            scores = model(g, e.cuda(), l.cuda())
            l = loss(scores, y.float().cuda())
            l.backward()
            optimizer.step()

def eval(model, dataset):
    return 0

def eval_kfold(drugs, protiens, relations, ddi, ppi, dpi, args):
    kf = KFold(n_splits=args.n_folds, shuffle=True)
    accuracy = []
    for train_ids, test_ids in kf.split(ddi):
        train_ddi = ddi.iloc[train_ids]
        test_ddi = ddi.iloc[test_ids]
        train_dataset = GraphDataset(drugs, protiens, relations, train_ddi, \
                                     ppi, dpi)
        test_dataset = GraphDataset(drugs, protiens, relations, test_ddi, \
                                     ppi, dpi)
        model = LinkPrediction(train_dataset.g).cuda()
        train(model, train_dataset, args)
        acc = eval(model, test_dataset)
        accuracy.append(acc)
    return np.mean(accuracy)



def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = parse_args()
    drugs, protiens,relations, ddi, ppi, dpi = read_data(args)
    dataset = GraphDataset(drugs, protiens,relations, ddi, ppi, dpi)
    model = LinkPrediction(dataset.g).cuda()
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(100):
        for i, (g, e, l, y) in tqdm(enumerate(dataset.get_batches(args.batch_size)),
                        desc='Epoch %d'%epoch,total=len(ddi)/args.batch_size):
            optimizer.zero_grad()
            scores = model(g, e.cuda(), l.cuda())
            l = loss(scores, y.float().cuda())
            l.backward()
            optimizer.step()
            # print(l.cpu().detach())



if __name__ == "__main__":
    main()
