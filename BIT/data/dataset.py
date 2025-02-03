from functools import lru_cache
import os
from ogb.lsc import PCQM4Mv2Evaluator
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import random_split
from fairseq.data import FairseqDataset, BaseWrapperDataset, LRUCacheDataset, data_utils

from .ligand_large_dataset import LargeLigandDataset
from .wrapper import MyPygPCQM4MDataset, MyPygPCQM4MPosDataset, MyPCQBLPPosDataset, convert_to_single_emb
from .collator import collator, collator_2d_moleculenet, collator_3d, collator_3d_qm9, collator_protein, \
    collator_pretrain, collator_complex, collator_complex_pdbbind, collator_complex_cls
from .ogb_datasets import OGBDatasetLookupTable
from .pyg_datasets import PYGDatasetLookupTable

class PCQPreprocessedData():
    def __init__(self, dataset_name, dataset_path="../dataset"):
        super().__init__()

        assert dataset_name in [
            "PCQM4M-LSC-V2",
            "PCQM4M-LSC-V2-TOY",
            "PCQM4M-LSC-V2-3D"
        ], "Only support PCQM4M-LSC-V2 or PCQM4M-LSC-V2-POS"
        self.dataset_name = dataset_name
        if dataset_name == 'PCQM4M-LSC-V2-3D':
            self.dataset = MyPygPCQM4MPosDataset(root=dataset_path)
        else:
            self.dataset = MyPygPCQM4MDataset(root=dataset_path)
        self.setup()

    def setup(self, stage: str = None):
        split_idx = self.dataset.get_idx_split()
        if self.dataset_name in ["PCQM4M-LSC-V2", "PCQM4M-LSC-V2-3D"]:
            self.train_idx = split_idx["train"]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test-dev"]
        elif self.dataset_name == "PCQM4M-LSC-V2-TOY":
            self.train_idx = split_idx["train"][:5000]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test-dev"]

        self.dataset_train = self.dataset.index_select(self.train_idx)
        self.dataset_val = self.dataset.index_select(self.valid_idx)
        self.dataset_test = self.dataset.index_select(self.test_idx)

        self.max_node = 256
        self.multi_hop_max_dist = 5 # max SPD
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class PCQBLPPreprocessedData():
    def __init__(self, dataset_name, dataset_path="../dataset"):
        super().__init__()
        type = "chnopsfcl500_re"
        assert dataset_name in [
            "PbCmQC-7M",
            "PbCmQC-79M",
            "PCQM4M-&-PbCmQC-7M",
            "PCQM4M-&-PbCmQC-79M",
            "PCQM4M-LSC-V2-3D",
            "PCQM4M-LSC-V2-3D-test",  # pcqm模型在新数据集上验证
            "PCQM4M-&-PbCmQC-79M-test"  # 新数据集模型在pcqm验证
        ], "Only support PubChemQC-B3LYP \ PCQM4M"
        self.dataset_name = dataset_name
        if dataset_name == 'PbCmQC-7M':
            self.dataset = MyPCQBLPPosDataset(root=dataset_path)
        elif dataset_name == 'PbCmQC-79M':
            self.dataset = LargeLigandDataset(root=dataset_path, type=type)
        elif dataset_name == "PCQM4M-&-PbCmQC-7M":
            self.dataset1 = MyPCQBLPPosDataset(root=dataset_path)   # 700w
            self.dataset2 = MyPygPCQM4MPosDataset(root=os.path.join(dataset_path, "molecules"))  # 300w
        elif dataset_name == "PCQM4M-&-PbCmQC-79M":
            self.dataset1 = LargeLigandDataset(root=dataset_path, type=type)   # 79M
            self.dataset2 = MyPygPCQM4MPosDataset(root=os.path.join(dataset_path, "molecules"))  # 3M
        elif dataset_name == "PCQM4M-LSC-V2-3D":
            self.dataset = MyPygPCQM4MPosDataset(root=os.path.join(dataset_path, "molecules"))  # 3M
        elif dataset_name == "PCQM4M-LSC-V2-3D-test":
            self.dataset = MyPygPCQM4MPosDataset(root=os.path.join(dataset_path, "molecules"))  # 3M
            self.dataset_valid = LargeLigandDataset(root=dataset_path, type="all_re")   # 79M
        elif dataset_name == "PCQM4M-&-PbCmQC-79M-test":
            self.dataset1 = LargeLigandDataset(root=dataset_path, type=type)   # 79M
            self.dataset2 = MyPygPCQM4MPosDataset(root=os.path.join(dataset_path, "molecules"))  # 3M
        self.setup()

    def setup(self, stage: str = None):
        # shuffle dataset
        if self.dataset_name == "PCQM4M-&-PbCmQC-7M":
            self.dataset1 = self.dataset1.shuffle()
            split_idx = self.dataset2.get_idx_split()
            self.dataset2 = self.dataset2.index_select(split_idx["train"])
            self.dataset2 = self.dataset2.shuffle()
            self.dataset_train = self.dataset1.index_select(range(50000, len(self.dataset1))) \
                                 + self.dataset2.index_select(range(50000, len(self.dataset2)))
            self.dataset_valid = self.dataset1.index_select(range(50000)) \
                                 + self.dataset2.index_select(range(50000))
            # self.dataset_train = self.dataset1 \
            #                      + self.dataset2.index_select(range(50000, len(self.dataset2)))
            # self.dataset_valid = self.dataset2.index_select(range(50000))

        elif self.dataset_name == 'PCQM4M-&-PbCmQC-79M':
            train_size = int(0.9993 * len(self.dataset1))
            val_size = len(self.dataset1) - train_size
            dataset_train1, dataset_valid1 = random_split(self.dataset1, [train_size, val_size])
            split_idx = self.dataset2.get_idx_split()
            self.dataset2 = self.dataset2.index_select(split_idx["train"])
            self.dataset2 = self.dataset2.shuffle()
            self.dataset_train = dataset_train1 + self.dataset2.index_select(range(50000, len(self.dataset2)))
            self.dataset_valid = dataset_valid1 + self.dataset2.index_select(range(50000))
            # self.dataset_train = self.dataset1 + self.dataset2.index_select(range(50000, len(self.dataset2)))
            # self.dataset_valid = self.dataset2.index_select(range(50000))

        elif self.dataset_name == 'PbCmQC-7M':
            self.dataset = self.dataset.shuffle()
            self.dataset_train = self.dataset.index_select(range(100000, len(self.dataset)))
            self.dataset_valid = self.dataset.indbex_select(range(100000))

        elif self.dataset_name == 'PbCmQC-79M':
            train_size = int(0.99 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.dataset_train, self.dataset_valid = random_split(self.dataset, [train_size, val_size])
        elif self.dataset_name == "PCQM4M-LSC-V2-3D":
            split_idx = self.dataset.get_idx_split()
            self.dataset = self.dataset.index_select(split_idx["train"])
            self.dataset = self.dataset.shuffle()
            self.dataset_train = self.dataset.index_select(range(50000, len(self.dataset)))
            self.dataset_valid = self.dataset.index_select(range(50000))

        elif self.dataset_name == "PCQM4M-LSC-V2-3D-test":
            split_idx = self.dataset.get_idx_split()
            self.dataset = self.dataset.index_select(split_idx["train"])
            self.dataset = self.dataset.shuffle()
            train_size = len(self.dataset_valid) - 50000
            val_size = 50000
            dataset_train1, dataset_valid1 = random_split(self.dataset_valid, [train_size, val_size])
            self.dataset_train = self.dataset.index_select(range(50000, len(self.dataset)))
            self.dataset_valid = dataset_valid1

        elif self.dataset_name == "PCQM4M-&-PbCmQC-79M-test":
            train_size = int(0.9993 * len(self.dataset1))
            val_size = len(self.dataset1) - train_size
            dataset_train1, dataset_valid1 = random_split(self.dataset1, [train_size, val_size])
            split_idx = self.dataset2.get_idx_split()
            self.dataset2 = self.dataset2.index_select(split_idx["train"])
            self.dataset2 = self.dataset2.shuffle()
            self.dataset_train = dataset_train1 + self.dataset2.index_select(range(50000, len(self.dataset2)))
            self.dataset_valid = self.dataset2.index_select(range(50000))


        self.max_node = 512
        self.multi_hop_max_dist = 5 # max SPD
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class PYGPreprocessedData():
    def __init__(self, dataset_name, dataset_path = "../dataset", seed=42, task_idx=4):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_name, seed=seed, dataset_path=dataset_path, task_idx=task_idx)
        self.setup()

    def setup(self, stage: str = None):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data

        self.max_node = 128
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class OGBPreprocessedData():
    def __init__(self, dataset_name, dataset_path="../dataset", seed=42):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset = OGBDatasetLookupTable.GetOGBDataset(
            dataset_name, seed=seed)
        self.setup()

    def setup(self, stage: str = None):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data
        self.max_node = 128
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = self.dataset.dataset.num_tasks
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),

class OGBPreprocessedDataTest():
    def __init__(self, dataset_name, dataset_path="../dataset", seed=42):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset = OGBDatasetLookupTable.GetOGBDataset(
            dataset_name, seed=seed)
        self.setup()

    def setup(self, stage: str = None):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.test_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.test_data
        self.dataset_test = self.dataset.test_data
        self.max_node = 128
        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.loss_fn = F.l1_loss
        self.num_class = self.dataset.dataset.num_tasks
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator(),


class MaskAtomDataset(BaseWrapperDataset):
    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs))
            # LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            # LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_atom_type, 
        num_edge_type, 
        seed: int = 1,
        mask_rate: float = 0.15, 
        mask_edge=True
    ):
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
        self.dataset = dataset
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.seed = seed
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, seed: int, epoch: int, index: int, masked_atom_indices=None):
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
        seed = int(hash((seed, epoch, index)) % 1e6)
        rng = np.random.default_rng(seed)
        data = self.dataset[index]

        assert (
            convert_to_single_emb(self.num_atom_type) not in data.x[:,0]
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            convert_to_single_emb(self.num_atom_type).item())

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            if hasattr(data, 'len_ligand'):
                num_atoms = data.len_ligand
            else:
                num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = rng.choice(num_atoms, sample_size, replace=False)
            masked_atom_indices.sort()

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        new_data = data.clone()
        new_data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        new_data.masked_atom_indices = torch.tensor(masked_atom_indices)
        new_data.raw_item.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            new_data.x[atom_idx] = convert_to_single_emb(torch.tensor(
                [[self.num_atom_type, 0, 12, 12, 10, 6, 6, 2, 2]]))

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

        return new_data

class MaskPocketDataset(BaseWrapperDataset):
    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, *args, **kwargs))
            # LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=False)),
            # LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)),
        )

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_atom_type, 
        num_edge_type, 
        seed: int = 1,
        mask_rate: float = 0.15, 
        mask_edge=True,
        mask_type='residue'
    ):
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
        self.dataset = dataset
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.seed = seed
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.mask_type = mask_type

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.seed, self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, seed: int, epoch: int, index: int, masked_atom_indices=None):
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
        seed = int(hash((seed, epoch, index)) % 1e6)
        rng = np.random.default_rng(seed)
        data = self.dataset[index]

        assert (
            convert_to_single_emb(self.num_atom_type) not in data.x[:,0]
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            convert_to_single_emb(self.num_atom_type).item())

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom

            # chain_idxs = data.chain_idxs
            # res_list = list(set(chain_idxs))
            # num_residues = len(res_list)
            # sample_size = int(num_residues * self.mask_rate + 1)
            # masked_res_indices = rng.choice(res_list, sample_size, replace=False)
            # masked_atom_indices = np.where(np.isin(chain_idxs, masked_res_indices)==True)[0]
            if self.mask_type == "residue":
                res_list = data.residues
                num_residues = len(res_list)
                sample_size = int(num_residues * self.mask_rate + 1)
                masked_res_indices = rng.choice(res_list, sample_size, replace=False)
                masked_atom_indices = np.array(sum([res['atoms'] for res in masked_res_indices],[]))
                masked_atom_indices.sort()
            elif self.mask_type == 'atom':
                num_atoms = data.x.size()[0]
                sample_size = int(num_atoms * self.mask_rate + 1)
                masked_atom_indices = rng.choice(num_atoms, sample_size, replace=False)
                masked_atom_indices.sort()
            else:
                raise TypeError(f"No such type: {self.mask_type}")
            
        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        new_data = data.clone()
        new_data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        new_data.masked_atom_indices = torch.tensor(masked_atom_indices)
        new_data.raw_item.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            new_data.x[atom_idx] = convert_to_single_emb(torch.tensor(
                [[self.num_atom_type, 0, 12, 12, 10, 6, 6, 2, 2]]))
                # [[self.num_atom_type, 20, 2]]))
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

        return new_data
    
class BatchedDataDataset(FairseqDataset):
    def __init__(self, dataset, dataset_version="2D", max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

        self.dataset_version = dataset_version
        assert self.dataset_version in ["2D", "3D", "pretrain",
                                        "3D_QM9", "protein", "complex",
                                        "complex_pdbbind", "2D_moleculenet",
                                        "complex_cls"]

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.dataset_version == '2D':
            collator_fn = collator
        elif self.dataset_version == '2D_moleculenet':
            collator_fn = collator_2d_moleculenet
        elif self.dataset_version == '3D':
            collator_fn = collator_3d
        elif self.dataset_version == '3D_QM9':
            collator_fn = collator_3d_qm9
        elif self.dataset_version == 'protein':
            collator_fn = collator_protein
        elif self.dataset_version == 'pretrain':
            collator_fn = collator_pretrain
        elif self.dataset_version == 'complex':
            collator_fn = collator_complex
        elif self.dataset_version == 'complex_cls':
            collator_fn = collator_complex_cls
        elif self.dataset_version == 'complex_pdbbind':
            collator_fn = collator_complex_pdbbind

        return collator_fn(samples,
                           max_node=self.max_node,
                           multi_hop_max_dist=self.multi_hop_max_dist,
                           spatial_pos_max=self.spatial_pos_max)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.max_node


class CacheAllDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.dataset[index]

    def collater(self, samples):
        return self.dataset.collater(samples)


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, size, seed):
        super().__init__(dataset)
        self.size = size
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.size)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)
