#!/usr/bin/env python

from argparser import parse_args
from model import LinkPrediction 
from tqdm import tqdm
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
    model = LinkPrediction(dataset.g).cuda()
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for epoch in range(100):
        for i, (g, e, l, y) in tqdm(enumerate(dataset.get_batches(256)),
                                    desc='Epoch %d'%epoch):
            optimizer.zero_grad()
            scores = model(g, e.cuda(), l.cuda())
            l = loss(scores, y.float().cuda())
            l.backward()
            optimizer.step()
            # print(l.cpu().detach())



if __name__ == "__main__":
    main()
