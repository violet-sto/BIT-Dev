import os
import pickle
import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch, InMemoryDataset
from tqdm.auto import tqdm
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem
from ogb.utils.torch_util import replace_numpy_with_torchtensor
# from oddt.toolkits.extras.rdkit.fixer import PreparePDBMol, PrepareComplexes, ExtractPocketAndLigand
from ...data_complex.protein_ligand import PDBProtein
from ...data_complex.data import ProteinLigandData, torchify_dict, complex2graph
from functools import lru_cache
from ..wrapper import mol2graph, preprocess_item, preprocess_complex
from prody import parsePDB, writePDB

class MyPDBbindDataset(InMemoryDataset):
    def __init__(self, root = 'dataset', view ='atom', transform=None, pre_transform = None):
        '''
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.root = root
        self.view = view
        
        super().__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y.mean().item()

    def std(self, idxs):
        y = torch.cat([self.get(i).y for i in idxs], dim=0)
        return y.std().item()

    @property
    def raw_file_names(self):
        self.core_path = os.path.join(self.raw_dir, 'CASF-2016', 'coreset')
        self.refined_path = os.path.join(self.raw_dir, 'refined-set')
        self.index_path = os.path.join(self.refined_path, 'index/INDEX_general_PL_data.2016')
        return sorted(os.listdir(self.raw_dir))
    
    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass

    def process(self):
        core_set_list = [x for x in os.listdir(self.core_path) if len(x) == 4]

        label = dict()
        with open(self.index_path, 'r') as f:
            for line in f:
                if '#' in line:
                    continue
                cont = line.strip().split()
                if len(cont) < 5:
                    continue
                code, pk = cont[0], cont[3]
                label[code] = float(pk) # -logKd/Ki

        num_skipped = 0
        refined_data = []
        core_data = []
        for i, name in enumerate(tqdm(os.listdir(self.refined_path))):
            if len(name) != 4: continue # 4057: 4059-2 (index and readme)
            try:
                ligand = Chem.MolFromMol2File('%s/%s/%s_ligand.mol2' % (self.refined_path, name, name))
                if ligand is None: # Some Mol2Files cannot be parsed
                    ligand = Chem.MolFromMolFile('%s/%s/%s_ligand.sdf' % (self.refined_path, name, name))

                protein = parsePDB('%s/%s/%s_protein.pdb' % (self.refined_path, name, name))
                pocket = protein.select('protein and not hydrogen and not hetatm and same residue as within 5 of ligand', ligand=ligand.GetConformer().GetPositions())
                writePDB('%s/%s/%s_pocket_5_prody.pdb' % (self.refined_path, name, name), pocket)

                ligand_dict = mol2graph(ligand)
                if self.view == 'residue':
                    receptor_dict = PDBProtein('%s/%s/%s_pocket_5_prody.pdb' % (self.refined_path, name, name)).to_dict_residue()
                    data = Data.from_dict(torchify_dict(ligand_dict))
                    data.receptor = Data.from_dict(torchify_dict(receptor_dict))
                    data.y = label[name]
                elif self.view == 'atom':
                    receptor_dict = PDBProtein('%s/%s/%s_pocket_5_prody.pdb' % (self.refined_path, name, name)).to_dict_atom()
                    data = complex2graph(torchify_dict(receptor_dict),torchify_dict(ligand_dict))
                    data.y = label[name]
                    
                if name in core_set_list:
                    core_data.append(data)
                    continue
                refined_data.append(data)

            except:
                num_skipped += 1
                print('Skipping (%d) %s' % (num_skipped, name, ))
                continue
        
        data, slices = self.collate(refined_data + core_data)

        train_idxs, valid_idxs = random_split(len(refined_data), split_ratio=0.9, seed=2020, shuffle=True)
        test_idxs = np.arange(len(core_data))

        print('Saving...')
        torch.save({'train':train_idxs,
                    'valid':valid_idxs,
                    'test':test_idxs + len(refined_data)}, os.path.join(self.root,'split_dict.pt'))
        torch.save((data, slices), self.processed_paths[0])
    
    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(os.path.join(self.root, 'split_dict.pt')))
        return split_dict

def random_split(dataset_size, split_ratio=0.9, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return np.array(train_idx), np.array(valid_idx)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    MyPDBbindDataset(args.path)
