# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import math
from collections import OrderedDict, defaultdict
import contextlib
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf
import shutil

import torch
from torch_geometric.transforms import Compose
from torch.utils.data import random_split
import numpy as np
from fairseq.data import (
    ConcatSentencesDataset,
    ConformerSampleDataset,
    ConformerSamplePocketDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    RoundRobinZipDatasets,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum
from fairseq.optim.amp_optimizer import AMPOptimizer

from ..data.dataset import (
    PCQPreprocessedData,
    OGBPreprocessedData,
    MaskAtomDataset,
    MaskPocketDataset,
    BatchedDataDataset,
    CacheAllDataset,
    EpochShuffleDataset,
    TargetDataset,
)
from ..data.wrapper import MaskAtom
from ..data.lmdb_dataset import LMDBDataset
from ..data.pocket_dataset import PocketDataset
from ..data_complex.datasets.pl import PocketLigandPairDataset
from ..data_complex.transforms import MyTruncateProtein

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class GraphPreTrainingConfig(FairseqDataclass):
    # data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    data_path: str = field(
        default="",
        metadata={
            "help": "path to data file"
        },
    )
    input_modality: str = field(
        default="complex,ligand,pocket",
        metadata={
            "help": "input modality"
        }
    )

    noise_scale: float = field(
        default=0.01,
        metadata={
            "help": "noise scale"
        },
    )

    sandwich_ln: bool = field(
        default=False,
        metadata={"help": "apply layernorm via sandwich form"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )
    init_token: Optional[int] = field(
        default=None,
        metadata={"help": "add token at the beginning of each batch item"},
    )
    separator_token: Optional[int] = field(
        default=None,
        metadata={"help": "add separator token between inputs"},
    )
    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    add_prev_output_tokens: bool = field(
        default=False,
        metadata={
            "help": "add prev_output_tokens to sample, used for encoder-decoder arch"
        },
    )
    max_positions: int = field(
        default=1024,
        metadata={"help": "max tokens per example"},
    )

    dataset_name: str = field(
        default="PCQM4M-LSC",
        metadata={"help": "name of the dataset"},
    )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    regression_target: bool = II("criterion.regression_target")
    # classification_head_name: str = II("criterion.classification_head_name")
    seed: int = II("common.seed")


@register_task("graph_pretraining", dataclass=GraphPreTrainingConfig)
class GraphPretrainingTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_modality = self.cfg.input_modality.split(',')

        if 'ligand' in self.input_modality:
            self.dm = PCQPreprocessedData(
                dataset_name=self.cfg.dataset_name, dataset_path=os.path.join(self.cfg.data_path, 'molecules'))
            self.dm.dataset_train = self.dm.dataset_train.shuffle()

        if 'pocket' in self.input_modality:
            self.pocket_dataset = PocketDataset(
                os.path.join(self.cfg.data_path, 'pockets/PDB_p2rank'))
            train_size = int(0.98 * len(self.pocket_dataset))
            val_size = len(self.pocket_dataset) - train_size
            self.pocket_train_dataset, self.pocket_val_dataset = random_split(self.pocket_dataset, [train_size, val_size])

        if 'complex' in self.input_modality:
            self.complex_dataset = PocketLigandPairDataset(
                os.path.join(self.cfg.data_path, 'complex/BioLiP2'), view='atom')
            train_size = int(0.9 * len(self.complex_dataset))
            val_size = len(self.complex_dataset) - train_size
            self.complex_train_dataset, self.complex_val_dataset = random_split(self.complex_dataset, [train_size, val_size])

    @ classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]
        modality_dataset = {}

        # Load dataset of ligands (from ogb)
        if split == "train":
            if 'ligand' in self.input_modality:
                ligand_batched_data = self.dm.dataset_train.index_select(range(100000, len(self.dm.dataset_train)))
                # ligand_batched_data = self.dm.dataset_train.index_select(range(1000000))
                ligand_batched_data = MaskAtomDataset.apply_mask(ligand_batched_data, 
                                                      num_atom_type = 119, 
                                                      num_edge_type = 5, 
                                                      mask_rate = 0.15, 
                                                      mask_edge = False)
                ligand_batched_data = BatchedDataDataset(ligand_batched_data,
                                                         dataset_version="pretrain",
                                                         max_node=self.dm.max_node,
                                                         multi_hop_max_dist=self.dm.multi_hop_max_dist,
                                                         spatial_pos_max=self.dm.spatial_pos_max
                                                         )
                ligand_batched_data = EpochShuffleDataset(
                    ligand_batched_data, size=len(ligand_batched_data), seed=self.cfg.seed)
                modality_dataset['ligand'] = ligand_batched_data

                logger.info("Loaded {0} with #ligand: {1}".format(
                    split, len(ligand_batched_data)))

            if 'pocket' in self.input_modality:
                pocket_batched_data = self.pocket_train_dataset
                pocket_batched_data = MaskPocketDataset.apply_mask(pocket_batched_data,
                                                        num_atom_type = 119, 
                                                        num_edge_type = 5, 
                                                        mask_rate = 0.15, 
                                                        mask_edge = False)
                pocket_batched_data = BatchedDataDataset(pocket_batched_data,
                                                         dataset_version="pretrain",
                                                         max_node=self.cfg.max_positions,
                                                         multi_hop_max_dist=self.pocket_dataset.multi_hop_max_dist,
                                                         spatial_pos_max=self.pocket_dataset.spatial_pos_max
                                                         )
                pocket_batched_data = EpochShuffleDataset(
                    pocket_batched_data, size=len(pocket_batched_data), seed=self.cfg.seed)
                modality_dataset['pocket'] = pocket_batched_data

                logger.info("Loaded {0} with #pocket: {1}".format(
                    split, len(pocket_batched_data)))


            if 'complex' in self.input_modality:
                complex_batched_data = self.complex_train_dataset
                complex_batched_data = MaskAtomDataset.apply_mask(complex_batched_data, 
                                                      num_atom_type = 119, 
                                                      num_edge_type = 5, 
                                                      mask_rate = 0.15, 
                                                      mask_edge = False)
                complex_batched_data = BatchedDataDataset(complex_batched_data,
                                                          dataset_version="complex_cls",
                                                          max_node=self.cfg.max_positions,
                                                          multi_hop_max_dist=self.complex_dataset.multi_hop_max_dist,
                                                          spatial_pos_max=self.complex_dataset.spatial_pos_max
                                                          )
                complex_batched_data = EpochShuffleDataset(
                    complex_batched_data, size=len(complex_batched_data), seed=self.cfg.seed)
                modality_dataset['complex'] = complex_batched_data

                logger.info("Loaded {0} with #complex: {1}".format(
                    split, len(complex_batched_data)))

            dataset = RoundRobinZipDatasets(
                OrderedDict([
                    (modality, modality_dataset[modality])
                    for modality in self.input_modality
                ])
            )

        elif split == "valid":
            if 'ligand' in self.input_modality:
                ligand_batched_data = self.dm.dataset_train.index_select(range(100000))
                ligand_batched_data = MaskAtomDataset.apply_mask(ligand_batched_data,
                                                                 num_atom_type=119,
                                                                 num_edge_type=5,
                                                                 mask_rate=0.15,
                                                                 mask_edge=False)
                ligand_batched_data = BatchedDataDataset(ligand_batched_data,
                                                         dataset_version="pretrain",
                                                         max_node=self.dm.max_node,
                                                         multi_hop_max_dist=self.dm.multi_hop_max_dist,
                                                         spatial_pos_max=self.dm.spatial_pos_max
                                                         )
                modality_dataset['ligand'] = ligand_batched_data

                logger.info("Loaded {0} with #ligand: {1}".format(
                    split, len(ligand_batched_data)))
            if 'pocket' in self.input_modality:
                pocket_batched_data = self.pocket_val_dataset
                pocket_batched_data = MaskPocketDataset.apply_mask(pocket_batched_data,
                                                        num_atom_type = 119,
                                                        num_edge_type = 5,
                                                        mask_rate = 0.15,
                                                        mask_edge = False)
                pocket_batched_data = BatchedDataDataset(pocket_batched_data,
                                                         dataset_version="pretrain",
                                                         max_node=self.cfg.max_positions,
                                                         multi_hop_max_dist=self.pocket_dataset.multi_hop_max_dist,
                                                         spatial_pos_max=self.pocket_dataset.spatial_pos_max
                                                         )
                modality_dataset['pocket'] = pocket_batched_data

                logger.info("Loaded {0} with #pocket: {1}".format(
                    split, len(pocket_batched_data)))
            if 'complex' in self.input_modality:
                complex_batched_data = self.complex_val_dataset
                complex_batched_data = MaskAtomDataset.apply_mask(complex_batched_data,
                                                                  num_atom_type=119,
                                                                  num_edge_type=5,
                                                                  mask_rate=0.15,
                                                                  mask_edge=False)
                complex_batched_data = BatchedDataDataset(complex_batched_data,
                                                          dataset_version="complex_cls",
                                                          max_node=self.cfg.max_positions,
                                                          multi_hop_max_dist=self.complex_dataset.multi_hop_max_dist,
                                                          spatial_pos_max=self.complex_dataset.spatial_pos_max
                                                          )
                modality_dataset['complex'] = complex_batched_data

                logger.info("Loaded {0} with #complex: {1}".format(
                    split, len(complex_batched_data)))

            dataset = RoundRobinZipDatasets(
                OrderedDict([
                    (modality, modality_dataset[modality])
                    for modality in self.input_modality
                ])
            )
        elif split == "test":
            pass

        try:
            self.datasets[split] = dataset
        except:
            pass

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        return model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(
            float)
        for modality in self.input_modality:
            if sample[modality] is None or len(sample[modality]) == 0:
                continue
            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss, sample_size, logging_output = criterion(
                        model, sample[modality], modality)
            agg_loss += loss
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{modality}:{k}"] += logging_output[k]

        if ignore_grad:
            agg_loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            try:
                optimizer.backward(agg_loss)
            except:
                raise ValueError(sample["complex"], agg_loss)
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(
            float)
        with torch.no_grad():
            for modality in self.input_modality:
                if sample[modality] is None or len(sample[modality]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model, sample[modality], modality)
                agg_loss += loss
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{modality}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
