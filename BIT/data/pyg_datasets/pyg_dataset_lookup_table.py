# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset, GraphormerPYGDatasetQM9
from .qm9 import newQM9, newHQM9
from .pdbbind import MyPDBbindDataset
import torch.distributed as dist


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyHQM9(newHQM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyHQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyHQM9, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, dataset_path: str, task_idx: int, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None
        train_idx = None
        valid_idx = None
        test_idx = None

        root = "dataset"
        qm9_data = False
        pdbbind_data = False
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
            qm9_data = True
        elif name == "qm9H":
            inner_dataset = MyHQM9(root=root)
            qm9_data = True
        elif name == "pdbbind_v2016":
            inner_dataset = MyPDBbindDataset(root=os.path.join(dataset_path, name))
            qm9_data = True
            split_idx = inner_dataset.get_idx_split()
            train_idx = split_idx["train"]
            valid_idx = split_idx["valid"]
            test_idx = split_idx["test"]
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)

        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                ) if not qm9_data else GraphormerPYGDatasetQM9(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        elif train_idx is not None:
            return GraphormerPYGDataset(
                    inner_dataset,
                    seed,
                    train_idx,
                    valid_idx,
                    test_idx,
                ) if not qm9_data else GraphormerPYGDatasetQM9(
                    inner_dataset,
                    seed,
                    train_idx,
                    valid_idx,
                    test_idx,
                )
        else:
            data_func = GraphormerPYGDataset if not qm9_data else GraphormerPYGDatasetQM9
            return (
                None
                if inner_dataset is None
                else data_func(inner_dataset, seed)
            )
