#!/usr/bin/env python

from argparser import parse_args
from torchviz import make_dot, make_dot_from_trace
from model import * 
from tqdm import tqdm
from data import GraphDataset
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os


def metrics(predictions, targets):
    # How to compute AP@50?
    return {'AUROC': roc_auc_score(targets, predictions),\
            'AUPRC': average_precision_score(targets, predictions)}

def aggregate(metrics):
    keys = metrics[0].keys()
    agg = {key:np.mean([x[key] for x in metrics]) for key in keys}
    return agg



def filter_ddi(ddi, mincount):
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
    ddi = filter_ddi(pd.read_csv(args.ddi), args.mincount)
    ppi = pd.read_csv(args.ppi,nrows=10000)
    dpi = pd.read_csv(args.dpi)
    if not args.use_protien:
        # delete all rows. keep column structure
        ppi = ppi[0:0]
        dpi = dpi[0:0]
    drugs = sorted(list(set(ddi['STITCH 1'].values.tolist() \
                     + ddi['STITCH 2'].values.tolist())))
    protiens = sorted(list(set(ppi['Gene 1'].values.tolist()\
                        + ppi['Gene 2'].values.tolist())))
    relations = sorted(list(set(ddi['Polypharmacy Side Effect'])))
    print(len(relations))
    return drugs, protiens,relations, ddi, ppi, dpi

def train(model, dataset, val_dataset, args):
    model.train()
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(args.num_epochs):
        model.train()
        losses = []
        for i, (g, e, l, y, paths) in tqdm(enumerate(dataset.get_batches(args.batch_size,npaths=args.npaths)),
                        desc='Epoch %d'%epoch,total=len(dataset)/args.batch_size):
            optimizer.zero_grad()
            scores = model(g, e.cuda(), l.cuda(),paths)
            # print(scores.cpu().detach())
            l = loss(scores, y.float().cuda())
            # dot = make_dot(l,
            #                         params=dict(model.named_parameters()))
            # dot.format = 'png'
            # dot.render('torchviz-sample')

            l.backward()
            optimizer.step()
            losses.append(l.detach().cpu().numpy())
            # print(losses[-1])
        print('Epoch %d:'%epoch)
        print('Epoch loss = %f'%np.mean(losses))
        print(eval(model, dataset.g, val_dataset, args))
        # print(eval(model, dataset.g, dataset, args))

def eval(model, train_g, dataset, args):
    model.eval()
    targets = []
    predictions = []
    for i, (g, e, l, y, paths) in \
            tqdm(enumerate(dataset.get_batches(args.batch_size,npaths=args.npaths)),
                        desc='Evaluating ',total=len(dataset)/args.batch_size):
        with torch.no_grad():
            # we use the graph from the training data to avoid information
            # leakage
            prediction = model(train_g, e.cuda(),l.cuda(),paths).detach()
            targets.append(y.detach())
            predictions.append(prediction)
    all_targets = torch.cat(targets).detach().cpu().numpy()
    all_predictions = torch.cat(predictions).detach().cpu().numpy()
    return metrics(all_predictions, all_targets)

def eval_kfold(drugs, protiens, relations, ddi, ppi, dpi, args):
    kf = KFold(n_splits=args.n_folds, shuffle=True)
    metrics = []
    for fold, (trainval_ids, test_ids) in enumerate(kf.split(ddi)):
        train_ids, val_ids = train_test_split(trainval_ids, train_size=0.8)
        train_ddi = ddi.iloc[train_ids]
        val_ddi = ddi.iloc[val_ids]
        test_ddi = ddi.iloc[test_ids]
        train_dataset = GraphDataset(drugs, protiens, relations, train_ddi, \
                                     ppi, dpi)
        val_dataset = GraphDataset(drugs, protiens, relations, val_ddi, \
                                     ppi, dpi)
        test_dataset = GraphDataset(drugs, protiens, relations, test_ddi, \
                                     ppi, dpi)
        if args.model=='RGCN':
            model = LinkPrediction(train_dataset.g).cuda()
        elif args.model=='Diffpool':
            model = LinkPredictionDiffpool(train_dataset.g).cuda()
        else:
            model = PathAttention(train_dataset.g).cuda()
        train(model, train_dataset, val_dataset, args)
        acc = eval(model,train_dataset.g, test_dataset, args)
        print(acc)
        torch.save(model.state_dict(), os.path.join(args.savepath,
                                                    'Fold_%d.pt'%fold))
        metrics.append(acc)
    return aggregate(metrics)



def eval_1fold(drugs, protiens, relations, ddi, ppi, dpi, args):
    # kf = KFold(n_splits=args.n_folds, shuffle=True)
    np.random.seed(0)
    train_val_ids, test_ids =train_test_split(np.arange(len(ddi)), train_size=0.8)
    train_ids, val_ids =train_test_split(train_val_ids, train_size=0.75)

    train_ddi = ddi.iloc[train_ids]
    val_ddi = ddi.iloc[val_ids]
    test_ddi = ddi.iloc[test_ids]
    train_dataset = GraphDataset(drugs, protiens, relations, train_ddi, \
                                 ppi, dpi)
    val_dataset = GraphDataset(drugs, protiens, relations, val_ddi, \
                                 ppi, dpi)
    test_dataset = GraphDataset(drugs, protiens, relations, test_ddi, \
                                 ppi, dpi)
    if args.model=='RGCN':
        model = LinkPrediction(train_dataset.g).cuda()
    elif args.model=='Diffpool':
        model = LinkPredictionDiffpool(train_dataset.g).cuda()
        # model = LinkPrediction(train_dataset.g).cuda()
    elif args.model == 'DiffpoolPA':
        model = LinkPredictionDiffpoolPA(train_dataset.g).cuda()
    else:
        model = PathAttention(train_dataset.g).cuda()
    print(model)
    train(model, train_dataset, val_dataset, args)
    acc = eval(model,train_dataset.g, test_dataset, args)
    print(acc)
    torch.save(model.state_dict(), os.path.join(args.savepath,
                                                args.savefile))
    return acc





def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = parse_args()
    drugs, protiens,relations, ddi, ppi, dpi = read_data(args)
    # print(eval_kfold(drugs, protiens, relations, ddi, ppi, dpi, args))
    print(eval_1fold(drugs, protiens, relations, ddi, ppi, dpi, args))
    # dataset = GraphDataset(drugs, protiens,relations, ddi, ppi, dpi)
    # model = LinkPrediction(dataset.g).cuda()
    # loss = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # for epoch in range(100):
    #     for i, (g, e, l, y) in tqdm(enumerate(dataset.get_batches(args.batch_size)),
    #                     desc='Epoch %d'%epoch,total=len(ddi)/args.batch_size):
    #         optimizer.zero_grad()
    #         scores = model(g, e.cuda(), l.cuda())
    #         l = loss(scores, y.float().cuda())
    #         l.backward()
    #         optimizer.step()
    #         # print(l.cpu().detach())



if __name__ == "__main__":
    main()
