# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
from rdkit import Chem
import pickle5 as pickle
import torch
from torch_geometric.data import Data
import numpy as np
import collections
from functools import lru_cache
import logging
from .wrapper import preprocess_item
logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

        self.max_node = 256 # 99%: 189
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.atomic_numbers = {'H': 0, 'C': 5, 'N': 6, 'O': 7, 'S': 15}  #? Se in HETATM
        self.aa_name_sym = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }
        self.aa_name_number = {
            k: i for i, (k, _) in enumerate(self.aa_name_sym.items())
        }

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        data['idx'] = idx
        return preprocess_item(pocket2graph(data, self.atomic_numbers, self.aa_name_number))

def pocket2graph(data, atomic_numbers, aa_name_number):
    # atoms, coordinates, meta_info, side, residue, pdbid
    
    atoms = torch.tensor([atomic_numbers[a[0]] for a in data['atoms']])
    assert len(data['atoms']) > 0
    assert len(data['coordinates']) == 1
    coordinates = data['coordinates'][0]
    assert len(atoms) == len(coordinates)
    residues = np.array(data['res_names'])
    chain_idxs = np.array(data['residue'])
    is_backbone = (torch.tensor(data['side']) == 0).long()

    # delete H atom of pocket
    is_H_protein = (atoms == 0)
    if is_H_protein.sum() > 0:
        not_H_protein = ~is_H_protein
        atoms = atoms[not_H_protein]
        residues = residues[not_H_protein]
        chain_idxs = chain_idxs[not_H_protein]
        is_backbone = is_backbone[not_H_protein]
        coordinates = coordinates[not_H_protein]

    # delete HETATM
    # is_HETATM_protein = np.array([aa not in aa_name_number.keys() for aa in residues])
    # if is_HETATM_protein.sum() > 0:
    #     not_HETATM_protein = ~is_HETATM_protein
    #     atoms = atoms[not_HETATM_protein]
    #     residues = residues[not_HETATM_protein]
    #     is_backbone = is_backbone[not_HETATM_protein]
    #     coordinates = coordinates[not_HETATM_protein]    

    # normalize
    coordinates = coordinates - coordinates.mean(axis=0)

    residues = torch.tensor([aa_name_number[aa] if aa in aa_name_number else 20 for aa in residues])

    graph = Data()
    graph.idx = data['idx']
    graph.x = torch.stack([atoms, residues, is_backbone], dim=-1)
    graph.pos = torch.from_numpy(coordinates).to(torch.float32)

    graph.edge_index = torch.tensor([[],[]],dtype=torch.long)
    graph.edge_attr = torch.tensor([],dtype=torch.long)

    graph.chain_idxs = chain_idxs

    return graph
