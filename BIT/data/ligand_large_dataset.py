import lmdb
import os
import h5py
import glob
from tqdm import tqdm
from rdkit import Chem
import pickle5 as pickle
import torch
from torch.utils.data import Dataset, random_split

from torch.nn import functional as F
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import collections
from functools import lru_cache
import logging
from .wrapper import preprocess_item
from ogb.lsc import PCQM4Mv2Evaluator

from ..data_complex.protein_ligand import PDBProtein
from ..data_complex.data import torchify_dict
from Bio.PDB import PDBParser, PDBIO
from prody import parsePDB, writePDB

logger = logging.getLogger(__name__)


class LargeLigandDataset(Dataset):
    def __init__(self, root, type="all", transform=None):
        file_name = None
        if type == "all":
            file_name = "merged_all.h5"
        elif type == "chno300_re":
            file_name = "merged_chno300_re.h5"
        elif type == "chno500_re":
            file_name = "merged_chno500_re.h5"
        elif type == "chnopsfcl500_re":
            file_name = "merged_chnopsfcl500_re.h5"
        elif type == "all_re":
            file_name = "merged_all_re.h5"
        else:
            raise TypeError(f"Pubchem: No such type: {type}")
        self.raw_path = os.path.join(root, 'PubChemQC-B3LYP')
        self.processed_path = os.path.join(self.raw_path, file_name)
        self.transform = transform
        self.h5 = None
        self.data_slices = None
        self.length = None

        if not os.path.exists(self.processed_path):
            self._process()

        # self.max_node = 256 # 99%: 189
        self.multi_hop_max_dist = 5
        self.max_node = 512
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),

    def _open_h5(self):
        assert self.h5 is None, 'A connection has already been opened.'
        self.h5 = h5py.File(self.processed_path, 'r')
        self.length = len(self.h5['atom_slice']) - 1
        self.atom_slice = self.h5['atom_slice']
        self.edge_slice = self.h5['edge_slice']
        self.token_slice = self.h5['token_slice']

    def _close_h5(self):
        self.h5.close()
        self.h5 = None
        self.keys = None

    def _process(self):
        print("no _process function")
        pass

    def __len__(self):
        if self.length is None:
            self._open_h5()
        return self.length

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.h5 is None:
            self._open_h5()
        tag = 1
        i = 0
        while tag == 1:
            try:
                x = self.h5['atom_feat'][self.atom_slice[idx]: self.atom_slice[idx + 1]]
                pos = self.h5['atom_pos'][self.atom_slice[idx]: self.atom_slice[idx + 1]]
                edge_attr = self.h5['edge_feat'][self.edge_slice[idx]: self.edge_slice[idx + 1]]
                edge_index = self.h5['edge_index'][self.edge_slice[idx]: self.edge_slice[idx + 1]]
                data = Data()
                homolumogap = 0.5  # 没有这个标签
                data.__num_nodes__ = int(x.shape[0])
                data.edge_index = torch.from_numpy(
                    edge_index.T.astype("int32")).to(torch.int64)  # (2, n_edge) -> (2, n_edge)
                data.edge_attr = torch.from_numpy(
                    edge_attr).to(torch.int64)
                data.x = torch.from_numpy(
                    x).to(torch.int64)
                data.y = torch.Tensor([homolumogap])
                data.pos = torch.from_numpy(
                    pos).to(torch.float32)
                tag = 0
            except:
                i += 1
                print(f"restart h5 file {i}")
                self.h5 = None
                self._open_h5()
        data.idx = idx
        if self.transform is not None:
            data = self.transform(data)
        return preprocess_item(data)

class LargeLigandDatasetRandomConf(Dataset):
    def __init__(self, root, type="all", transform=None):
        file_name = None
        if type == "all":
            file_name = "merged_all.h5"
        elif type == "chno300_re":
            file_name = "merged_chno300_re.h5"
        elif type == "chno500_re":
            file_name = "merged_chno500_re.h5"
        elif type == "chnopsfcl500_re":
            file_name = "merged_chnopsfcl500_re.h5"
        elif type == "all_re":
            file_name = "merged_all_re.h5"
        else:
            raise TypeError(f"Pubchem: No such type: {type}")
        self.raw_path = os.path.join(root, 'PubChemQC-B3LYP')
        self.processed_path = os.path.join(self.raw_path, file_name)
        self.transform = transform
        self.h5 = None
        self.data_slices = None
        self.length = None

        if not os.path.exists(self.processed_path):
            self._process()

        # self.max_node = 256 # 99%: 189
        self.multi_hop_max_dist = 5
        self.max_node = 512
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),

    def _open_h5(self):
        assert self.h5 is None, 'A connection has already been opened.'
        self.h5 = h5py.File(self.processed_path, 'r')
        self.length = len(self.h5['atom_slice']) - 1
        self.atom_slice = self.h5['atom_slice']
        self.pos_slice = self.h5['pos_slice']
        self.edge_slice = self.h5['edge_slice']
        self.token_slice = self.h5['token_slice']

    def _close_h5(self):
        self.h5.close()
        self.h5 = None
        self.keys = None

    def _process(self):
        print("no _process function")
        pass

    def __len__(self):
        if self.length is None:
            self._open_h5()
        return self.length

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.h5 is None:
            self._open_h5()
        tag = 1
        i = 0
        while tag == 1:
            try:
                x = self.h5['atom_feat'][self.atom_slice[idx]: self.atom_slice[idx + 1]]
                pos = self.h5['atom_pos'][self.pos_slice[idx]: self.pos_slice[idx + 1]]
                edge_attr = self.h5['edge_feat'][self.edge_slice[idx]: self.edge_slice[idx + 1]]
                edge_index = self.h5['edge_index'][self.edge_slice[idx]: self.edge_slice[idx + 1]]
                n_atom = x.shape[0]
                conf_num = int(pos.shape[0] / n_atom)
                random_sample = np.random.randint(0, conf_num)
                pos_sample = pos[n_atom*random_sample: n_atom*(random_sample + 1), :]
                data = Data()
                homolumogap = 0.5  # 没有这个标签
                data.__num_nodes__ = int(x.shape[0])
                data.edge_index = torch.from_numpy(
                    edge_index.T.astype("int32")).to(torch.int64)  # (2, n_edge) -> (2, n_edge)
                data.edge_attr = torch.from_numpy(
                    edge_attr).to(torch.int64)
                data.x = torch.from_numpy(
                    x).to(torch.int64)
                data.y = torch.Tensor([homolumogap])
                data.pos = torch.from_numpy(
                    pos_sample).to(torch.float32)
                tag = 0
            except:
                i += 1
                print(f"restart h5 file {i}")
                self.h5 = None
                self._open_h5()
        data.idx = idx
        if self.transform is not None:
            data = self.transform(data)
        return preprocess_item(data)