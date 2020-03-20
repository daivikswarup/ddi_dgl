#!/usr/bin/env python
import argparse
def parse_args():
    """TODO: Docstring for parse_args.
    :returns: TODO

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ddi', action='store', dest='ddi',\
                help='Path to ddi csv',\
                default='../data/bio-decagon-combo.csv')
    parser.add_argument('-ppi', action='store', dest='ppi',\
                help='Path to ppi csv',\
                default='../data/bio-decagon-ppi.csv')
    parser.add_argument('-dpi', action='store', dest='dpi',\
                help='Path to dpi csv',\
                default='../data/bio-decagon-targets-all.csv')
    parser.add_argument('-n_folds', action='store', type=int,\
                help = 'Number of folds in training',\
                default = 10)
    parser.add_argument('-batch_size', action='store', type=int,\
                help = 'Batch_size',\
                default = 256)
    parser.add_argument('-num_epochs', action='store', type=int,\
                help = 'num_epochs',\
                default = 3)
    return parser.parse_args()

