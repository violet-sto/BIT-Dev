# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, math

import contextlib
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf

import torch
import numpy as np
from fairseq.data import (
    ConcatSentencesDataset,
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
    PYGPreprocessedData,
    OGBPreprocessedData,
    BatchedDataDataset,
    CacheAllDataset,
    EpochShuffleDataset,
    TargetDataset,
)

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    # data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    data_path: str = field(
        default="",
        metadata={
            "help": "path to data file"
        },
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
        default=512,
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

@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dm = PCQPreprocessedData(dataset_name=self.cfg.dataset_name
                                      , dataset_path=os.path.join(self.cfg.data_path, 'molecules'))

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(batched_data,
            dataset_version="3D",
            max_node=self.dm.max_node,
            multi_hop_max_dist=self.dm.multi_hop_max_dist,
            spatial_pos_max=self.dm.spatial_pos_max
        )
        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_data
            },
            "target": target
        }, sizes=np.array([1] * len(batched_data))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=self.cfg.seed)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        return model

    #
    # def train_step(
    #     self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    # ):
    #     """
    #     Do forward and backward, and return the loss as computed by *criterion*
    #     for the given *model* and *sample*.
    #
    #     Args:
    #         sample (dict): the mini-batch. The format is defined by the
    #             :class:`~fairseq.data.FairseqDataset`.
    #         model (~fairseq.models.BaseFairseqModel): the model
    #         criterion (~fairseq.criterions.FairseqCriterion): the criterion
    #         optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
    #         update_num (int): the current update
    #         ignore_grad (bool): multiply loss by 0 if this is set to True
    #
    #     Returns:
    #         tuple:
    #             - the loss
    #             - the sample size, which is used as the denominator for the
    #               gradient
    #             - logging outputs to display while training
    #     """
    #     model.train()
    #     model.set_num_updates(update_num)
    #     with torch.autograd.profiler.record_function("forward"):
    #         with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
    #             loss, sample_size, logging_output = criterion(model, sample, modality="ligand")
    #     if ignore_grad:
    #         loss *= 0
    #     with torch.autograd.profiler.record_function("backward"):
    #         optimizer.backward(loss)
    #     return loss, sample_size, logging_output
    #
    # def valid_step(self, sample, model, criterion):
    #     model.eval()
    #     with torch.no_grad():
    #         loss, sample_size, logging_output = criterion(model, sample, modality='ligand')
    #     return loss, sample_size, logging_output

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
    

@dataclass
class GraphPredictionConfigQM9(FairseqDataclass):
    # data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    data_path: str = field(
        default="",
        metadata={
            "help": "path to data file"
        },
    )

    loss_type: str = field(
        default="L1",
        metadata={
            "help": "loss type, L1 or L2"
        }
    )

    std_type: str = field(
        default="no_std",
        metadata={
            "help": "standardization type, no_std or std_labels or std_logits"
        }
    )

    readout_type: str = field(
        default="cls",
        metadata={
            "help": "readout function, cls or sum or mean"
        }
    )

    # qm9 task
    task_idx: int = field(
        default=0,
        metadata={"help": "qm9 task index (range from 0 to 18)"}
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
        default=512,
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

@register_task("graph_prediction_qm9", dataclass=GraphPredictionConfigQM9)
class GraphPredictionTaskQM9(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dm = PYGPreprocessedData(dataset_name=self.cfg.dataset_name, dataset_path=self.cfg.data_path, seed=self.cfg.seed, task_idx=self.cfg.task_idx)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(batched_data,
            dataset_version="3D_QM9",
            max_node=self.dm.max_node,
            multi_hop_max_dist=self.dm.multi_hop_max_dist,
            spatial_pos_max=self.dm.spatial_pos_max
        )

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_data
            },
            "target": target
        }, sizes=np.array([1] * len(batched_data))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=self.cfg.seed)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        return model

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


@dataclass
class GraphPredictionConfigPDBbind(FairseqDataclass):
    # data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    data_path: str = field(
        default="",
        metadata={
            "help": "path to data file"
        },
    )

    loss_type: str = field(
        default="L1",
        metadata={
            "help": "loss type, L1 or L2"
        }
    )

    std_type: str = field(
        default="no_std",
        metadata={
            "help": "standardization type, no_std or std_labels or std_logits"
        }
    )

    readout_type: str = field(
        default="cls",
        metadata={
            "help": "readout function, cls or sum or mean"
        }
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
        default=512,
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

@register_task("graph_prediction_pdbbind", dataclass=GraphPredictionConfigPDBbind)
class GraphPredictionTaskPDBbind(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dm = PYGPreprocessedData(dataset_name=self.cfg.dataset_name, dataset_path=os.path.join(self.cfg.data_path,'complex'), seed=self.cfg.seed)
        
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(batched_data,
            dataset_version="complex_pdbbind",
            max_node=1024,
            multi_hop_max_dist=self.dm.multi_hop_max_dist,
            spatial_pos_max=self.dm.spatial_pos_max
        )

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_data
            },
            "target": target
        }, sizes=np.array([1] * len(batched_data))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=self.cfg.seed)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        return model

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
    

@dataclass
class GraphPredictionWithFlagConfigPDBbind(GraphPredictionConfigPDBbind):
    flag_m: int = field(
        default=4,
        metadata={
            "help": "number of iterations to optimize the perturbations with flag objectives"
        },
    )

    flag_step_size: float = field(
        default=1e-2,
        metadata={
            "help": "learing rate of iterations to optimize the perturbations with flag objective"
        },
    )

    flag_mag: float = field(
        default=1e-2,
        metadata={"help": "magnitude bound for perturbations in flag objectives"},
    )


@register_task("graph_prediction_pdbbind_with_flag", dataclass=GraphPredictionWithFlagConfigPDBbind)
class GraphPredictionWithFlagTask(GraphPredictionTaskPDBbind):
    """
    Graph prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.flag_m = cfg.flag_m
        self.flag_step_size = cfg.flag_step_size
        self.flag_mag = cfg.flag_mag

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.
        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        batched_data = sample["net_input"]["batched_data"]["x"]
        n_graph, n_node = batched_data.shape[:2]
        perturb_shape = n_graph, n_node, model.encoder_embed_dim
        if self.flag_mag > 0:
            perturb = (
                torch.FloatTensor(*perturb_shape)
                .uniform_(-1, 1)
                .to(batched_data.device)
            )
            perturb = perturb * self.flag_mag / math.sqrt(perturb_shape[-1])
        else:
            perturb = (
                torch.FloatTensor(*perturb_shape)
                .uniform_(-self.flag_step_size, self.flag_step_size)
                .to(batched_data.device)
            )
        perturb.requires_grad_()
        sample["perturb"] = perturb
        with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
            loss, sample_size, logging_output = criterion(
                model, sample
            )
            if ignore_grad:
                loss *= 0
        loss /= self.flag_m
        total_loss = 0
        for _ in range(self.flag_m - 1):
            optimizer.backward(loss)
            total_loss += loss.detach()
            perturb_data = perturb.detach() + self.flag_step_size * torch.sign(
                perturb.grad.detach()
            )
            if self.flag_mag > 0:
                perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
                exceed_mask = (perturb_data_norm > self.flag_mag).to(perturb_data)
                reweights = (
                    self.flag_mag / perturb_data_norm * exceed_mask
                    + (1 - exceed_mask)
                ).unsqueeze(-1)
                perturb_data = (perturb_data * reweights).detach()
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            sample["perturb"] = perturb
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(
                    model, sample
                )
                if ignore_grad:
                    loss *= 0
            loss /= self.flag_m
        optimizer.backward(loss)
        total_loss += loss.detach()
        logging_output["loss"] = total_loss
        return total_loss, sample_size, logging_output