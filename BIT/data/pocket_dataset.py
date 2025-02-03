import lmdb
import os
import glob
from tqdm import tqdm
from rdkit import Chem
import pickle5 as pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import collections
from functools import lru_cache
import logging
from .wrapper import preprocess_item
from ..data_complex.protein_ligand import PDBProtein
from ..data_complex.data import torchify_dict
from Bio.PDB import PDBParser, PDBIO
from prody import parsePDB, writePDB
logger = logging.getLogger(__name__)


class PocketDataset(Dataset):
    def __init__(self, raw_path, transform=None):
        self.raw_path = raw_path.rstrip('/')
        self.processed_path = self.raw_path + '_processed_1110.lmdb'
        
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process()

        # env = self._connect_db()
        # with env.begin() as txn:
        #     self._keys = list(txn.cursor().iternext(values=False))

        # self.max_node = 256 # 99%: 189
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        # self.atomic_numbers = {'H': 0, 'C': 5, 'N': 6, 'O': 7, 'S': 15}  #? Se in HETATM
        # self.aa_name_sym = {
        #     'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        #     'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        #     'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        # }
        # self.aa_name_number = {
        #     k: i for i, (k, _) in enumerate(self.aa_name_sym.items())
        # }

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=50*(1024*1024*1024),   # 50GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,   
            readahead=True,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    # def _process(self):
    #     db = lmdb.open(
    #         self.processed_path,
    #         map_size=30*(1024*1024*1024),   # 30GB
    #         create=True,
    #         subdir=False,
    #         readonly=False, # Writable
    #     )

    #     receptor_list = os.listdir(self.raw_path)

    #     num_skipped = 0
    #     idx = 0
    #     with db.begin(write=True, buffers=True) as txn:
    #         for i, receptor_fn in enumerate(tqdm(receptor_list)):
    #             if receptor_fn is None: continue
    #             receptor = parsePDB(os.path.join(self.raw_path, receptor_fn, f'{receptor_fn}.pdb'))
    #             receptor = receptor.select('protein and not hydrogen and not hetatm')
    #             pockets_fn = os.path.join(self.raw_path, receptor_fn, 'pockets')
    #             pocket_list = [x for x in os.listdir(pockets_fn) if x.endswith('atm.pdb')]
    #             for pocket_fn in pocket_list:
    #                 try:
    #                     residue_dict = parse_pocket(os.path.join(pockets_fn, pocket_fn))
    #                     residues = []
    #                     for chain, resnum_list in residue_dict.items():
    #                         resnum_str = ' '.join(resnum_list)
    #                         residues.append('chain {} resnum {}'.format(chain, resnum_str))
                        
    #                     pocket = receptor.select(' or '.join(residues))
    #                     writePDB(os.path.join(pockets_fn, pocket_fn.replace('atm','prody')), pocket)

    #                     pocket_dict = PDBProtein(os.path.join(pockets_fn, pocket_fn.replace('atm','prody'))).to_dict_atom()
    #                     data = Data.from_dict(torchify_dict(pocket_dict))

    #                     txn.put(
    #                         key = str(idx).encode(),
    #                         value = pickle.dumps(data)
    #                     )
    #                     idx += 1
    #                 except:
    #                     num_skipped += 1
    #                     print('Skipping (%d) %s %s' % (num_skipped, receptor_fn, pocket_fn))
    #                     continue
 
    #     db.close()

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=50*(1024*1024*1024),   # 50GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )

        receptor_list = glob.glob(os.path.join(self.raw_path, '*pdb_predictions*'))

        num_skipped = 0
        idx = 0
        with db.begin(write=True, buffers=True) as txn:
            for pred_fn in tqdm(receptor_list):
                if pred_fn is None: continue
                df = pd.read_csv(pred_fn, usecols=['  rank',' residue_ids'])
                if len(df) == 0: continue

                code = os.path.basename(pred_fn).split('.')[0]
                receptor_fn = os.path.join(self.raw_path, 'visualizations', 'data', f'{code}.pdb')
                receptor = parsePDB(receptor_fn)
                receptor = receptor.select('protein and not hydrogen and not hetatm')

                for i, (rank, residue_ids) in tqdm(df.iterrows(), disable=True):
                    try:
                        residue_dict = parse_pocket(residue_ids)
                        residues = []
                        for chain, resnum_list in residue_dict.items():
                            resnum_str = ' '.join(resnum_list)
                            residues.append('chain {} resnum {}'.format(chain, resnum_str))
                        
                        pocket = receptor.select(' or '.join(residues))
                        pocket_fn = os.path.join(receptor_fn.replace('pdb',f'pocket{rank}.pdb'))
                        writePDB(pocket_fn, pocket)

                        pocket_dict = PDBProtein(pocket_fn).to_dict_atom()
                        data = Data.from_dict(torchify_dict(pocket_dict))

                        txn.put(
                            key = str(idx).encode(),
                            value = pickle.dumps(data)
                        )
                        idx += 1
                    except:
                        num_skipped += 1
                        print('Skipping (%d) %s %s' % (num_skipped, receptor_fn, pocket_fn))
                        continue
 
        db.close()

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        # data = pickle.loads(self.db.begin().get(key))
        tag = 1
        i = 0
        while tag == 1:
            try:
                with self.db.begin() as db:
                    data = pickle.loads(db.get(key))
                tag = 0
            except:
                i += 1
                print(key, i)
                self.db = None
                self._connect_db()
        data.idx = idx
        if self.transform is not None:
            data = self.transform(data)
        return preprocess_item(data)

# def parse_pocket(file):
#     residue_dict = collections.defaultdict(set)
#     with open(file, 'r') as f:
#         for line in f:
#             if line.startswith('ATOM'):
#                 chain_id = line[21]
#                 residue_num = line[22:27].strip()
#                 residue_dict[chain_id].add(residue_num)
#     return residue_dict

def parse_pocket(residue_ids):
    residue_list = residue_ids.strip().split(' ')
    residue_dict = collections.defaultdict(set)
    
    for item in residue_list:
        chain_id, residue_num = item.split('_')
        residue_dict[chain_id].add(residue_num)

    return residue_dict