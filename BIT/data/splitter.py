import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

# splitter function

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

# # test generate_scaffold
# s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
# scaffold = generate_scaffold(s)
# assert scaffold == 'c1ccc(Oc2ccccn2)cc1'

def scaffold_split(dataset, frac_train=None, frac_valid=None, frac_test=None):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds

    Args:
        dataset(InMemoryDataset): the dataset to split. Make sure each element in
            the dataset has key "smiles" which will be used to calculate the 
            scaffold.
        frac_train(float): the fraction of data to be used for the train split.
        frac_valid(float): the fraction of data to be used for the valid split.
        frac_test(float): the fraction of data to be used for the test split.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i in range(N):
        scaffold = generate_scaffold(dataset[i]['smiles'], include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]
    test_dataset = dataset[test_idx]
    return train_dataset, valid_dataset, test_dataset