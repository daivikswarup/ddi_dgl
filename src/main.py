#!/usr/bin/env python

from argparser import parse_args
from model import LinkPrediction 
from data import GraphDataset
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    args = parse_args()
    dataset = GraphDataset(args.ddi, args.ppi, args.dpi)
    model = LinkPrediction(dataset.g)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(100):
        for g, e, l, y in dataset.get_batches():
            optimizer.zero_grad()
            scores = model(g, e, l)
            l = loss(scores, y.float())
            l.backward()
            optimizer.step()
            print(l)



if __name__ == "__main__":
    main()
