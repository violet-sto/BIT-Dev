# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tarfile
from rdkit import Chem
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from multiprocessing import Pool
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pandas as pd
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.utils.torch_util import replace_numpy_with_torchtensor
import os.path as osp
import shutil
import os
import algos
import numpy as np
import random
from functools import lru_cache
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        # graph['edge_feat'] = edge_attr
        # graph['node_feat'] = x
        graph['edge_attr'] = edge_attr
        graph['x'] = x
        graph['num_nodes'] = len(x)

        return graph
    except:
        return None


def mol2graph(mol):
    try:

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        # positions
        positions = mol.GetConformer().GetPositions().astype(np.float32)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_attr'] = edge_attr
        graph['x'] = x
        graph['num_nodes'] = len(x)
        graph['pos'] = positions

        return graph
    except:
        return None


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(
            self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) ==
                            graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(
                        graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(
                        graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(
                        graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])

                    data_list.append(data)
                except:
                    continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.pos_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2PosDataset, self).__init__(
            self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, 'r:gz')
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print('Stop download')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        graph_pos_list = Chem.SDMolSupplier(
            osp.join(self.original_root, 'pcqm4m-v2-train.sdf'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smiles_list)

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) ==
                            graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(
                        graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(
                        graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(
                        graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.zeros(
                        data.__num_nodes__, 3).to(torch.float32)

                    data_list.append(data)
                except:
                    continue

        print('Extracting 3D positions from SDF files for Training Data...')
        train_data_with_position_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(mol2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
                try:
                    data = Data()
                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_attr']) ==
                            graph['edge_index'].shape[1])
                    assert (len(graph['x']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(
                        graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(
                        graph['edge_attr']).to(torch.int64)
                    data.x = torch.from_numpy(
                        graph['x']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(
                        graph['pos']).to(torch.float32)

                    train_data_with_position_list.append(data)
                except:
                    continue
        data_list = train_data_with_position_list + \
            data_list[len(train_data_with_position_list):]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


class PygPubchemQCPosDataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PubChemQC-B3LYP  dataset object
                - root (str): the dataset folder will be located at root
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'PubChemQC-B3LYP')
        self.version = 1

        super(PygPubchemQCPosDataset, self).__init__(
            self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return '1_10000000_final.sdf'

    @property
    def processed_file_names(self):
        return 'processed_1_10000000.pt'

    def download(self):
        raise ValueError("no data source")

    def process(self):
        graph_pos_list = Chem.SDMolSupplier(
            osp.join(self.raw_dir, '1_10000000_final.sdf'))

        data_list = []
        print('Extracting 3D positions from SDF files for Training Data...')
        train_data_with_position_list = []
        with Pool(processes=120) as pool:
            iter = pool.imap(mol2graph, graph_pos_list)

            for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
                try:
                    data = Data()
                    homolumogap = 0.5

                    assert (len(graph['edge_attr']) ==
                            graph['edge_index'].shape[1])
                    assert (len(graph['x']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(
                        graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(
                        graph['edge_attr']).to(torch.int64)
                    data.x = torch.from_numpy(
                        graph['x']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(
                        graph['pos']).to(torch.float32)

                    train_data_with_position_list.append(data)
                except:
                    continue
        data_list = train_data_with_position_list

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item, num_cls=1):
    raw_item = item.clone()
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x.to(torch.int64)
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    # max dist = 510 for disconnected graph
    # all -1 for disconnected graph
    # max_dist = np.amax(shortest_path_result)
    max_dist = max(5, np.amax(shortest_path_result[shortest_path_result != 510]))
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + num_cls, N + num_cls], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()
    item.raw_item = raw_item

    return item


def preprocess_complex(item):
    edge_attr, edge_index, x, receptor_x = item.edge_attr, item.edge_index.to(
        torch.int64), item.x.to(torch.int64), item.receptor.x.to(torch.int64)
    n = x.size(0)
    m = receptor_x.size(0)
    N = n+m
    # C:6, N:7, O:8; residue: 128+
    receptor_x_expand = F.pad(receptor_x + 128, (0, x.size(1)-1))
    x = convert_to_single_emb(torch.cat([x, receptor_x_expand], dim=0))

    # node adj matrix [N, N] bool
    adj = torch.zeros([n, n], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    # max dist = 510 for disconnected graph
    # all -1 for disconnected graph
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    edge_input_expand = -1 * np.ones((N, N, max_dist, edge_attr.size(-1)))
    edge_input_expand[:n, :n] = edge_input

    spatial_pos = -1 * np.ones([N, N], dtype=np.int64)
    spatial_pos[:n,:n] = shortest_path_result
    spatial_pos = torch.from_numpy((spatial_pos)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input_expand).long()
    item.len_ligand = n
    item.len_pocket = m

    return item


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            if hasattr(data, 'len_ligand'):
                num_atoms = data.len_ligand
            else:
                num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
            masked_atom_indices.sort()

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor(
                [self.num_atom_type, 0, 12, 12, 10, 6, 6, 2, 2])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


class MyPygPCQM4MPosDataset(PygPCQM4Mv2PosDataset):
    def download(self):
        super(MyPygPCQM4MPosDataset, self).download()

    def process(self):
        super(MyPygPCQM4MPosDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        """
        Pytorch Geometric data object
        """
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)

class MyPCQBLPPosDataset(PygPubchemQCPosDataset):
    def download(self):
        super(MyPCQBLPPosDataset, self).download()

    def process(self):
        super(MyPCQBLPPosDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        """
        Pytorch Geometric data object
        """
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


if __name__ == "__main__":
    dataset = PygPCQM4Mv2PosDataset()
    print(len(dataset))