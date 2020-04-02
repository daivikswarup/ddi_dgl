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
    parser.add_argument('-model', action='store', dest='model',\
                help='model to use',\
                default='pathattention')
    parser.add_argument('-savepath', action='store', dest='savepath',\
                help='Directory to save checkpoints',\
                default='../models')
    parser.add_argument('-npaths', action='store', type=int,\
                help = 'Number of paths to attend over',\
                default = 10)
    parser.add_argument('-n_folds', action='store', type=int,\
                help = 'Number of folds in training',\
                default = 10)
    parser.add_argument('-batch_size', action='store', type=int,\
                help = 'Batch_size',\
                default = 128)
    parser.add_argument('-num_epochs', action='store', type=int,\
                help = 'num_epochs',\
                default = 10)
    parser.add_argument('-mincount', action='store', type=int,\
                help = 'minimum number of occurences in the data',\
                default = 20000)
    parser.add_argument('-no_protiens', action='store_false', dest='use_protien',
                        help='do not use ppi, dpi edges')
    parser.add_argument('-use_protiens', action='store_true', dest='use_protien',
                        help='use ppi, dpi edges')
    parser.set_defaults(use_protien=True)

    return parser.parse_args()

