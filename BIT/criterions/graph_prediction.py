from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
from torch.nn import functional
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os


@dataclass
class GraphPredictionConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")

@dataclass
class GraphFinetuneConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.noise_scale = task.cfg.noise_scale

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        # add gaussian noise
        ori_pos = sample['net_input']['batched_data']['pos']
        noise = torch.randn(ori_pos.shape).to(ori_pos) * self.noise_scale
        noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        noise = noise.masked_fill_(noise_mask, 0.0)
        sample['net_input']['batched_data']['pos'] = ori_pos + noise


        # model_output = model(sample["net_input"]['batched_data'], modality="ligand")
        model_output = model(**sample["net_input"])
        logits, node_output = model_output[0], model_output[1]
        # logits = logits[:,0,:].reshape(-1)
        logits = logits[:,0,:]
        targets = model.get_targets(sample, [logits])

        loss = nn.L1Loss(reduction='sum')(logits, targets)

        if node_output is not None:
            node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + noise_mask
            node_output = node_output.masked_fill_(node_mask, 0.0)

            node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
            node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)

            tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
            tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            node_output_loss = (node_output_loss / tgt_count).sum() * 1
        else:
            node_output_loss = (noise - noise).sum()

        logging_output = {
            "loss": loss.data,
            "node_output_loss": node_output_loss.data,
            "total_loss": loss.data + node_output_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss + node_output_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        node_output_loss_sum = sum(log.get("node_output_loss", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "node_output_loss", node_output_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "total_loss", total_loss_sum / sample_size, sample_size, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_prediction_qm9", dataclass=GraphPredictionConfig)
class GraphPredictionLossQM9(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.loss_type = task.cfg.loss_type
        self.std_type = task.cfg.std_type
        self.readout_type = task.cfg.readout_type

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        logits = model(**sample["net_input"], modality='ligand')[0]
        logits_padding = sample['net_input']['batched_data']['x'][:, :, 0].eq(0).unsqueeze(-1)

        if self.readout_type == 'cls':
            logits = logits[:, 0, :]
        elif self.readout_type == 'mean':
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).mean(dim=1)
        else:
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).sum(dim=1)

        targets = model.get_targets(sample, [logits])

        mean = sample['net_input']['batched_data']['mean']
        std = sample['net_input']['batched_data']['std']

        if self.std_type == 'std_labels':
            new_labels = (targets - mean) / std
            new_logits = logits
        elif self.std_type == 'std_logits':
            new_labels = targets
            new_logits = logits * std + mean
        else:
            new_labels = targets
            new_logits = logits

        if self.std_type == 'no_std':
            loss_l1 = nn.L1Loss(reduction="sum")(logits, targets)
        else:
            loss_l1 = nn.L1Loss(reduction="sum")(logits * std + mean, targets)

        if sample['net_input']['batched_data']['type'] == 'train':
            loss_func = nn.L1Loss if self.loss_type == 'L1' else nn.MSELoss
            loss = loss_func(reduction="sum")(new_logits, new_labels)
        else:
            loss = loss_l1

        logging_output = {
            "loss": loss.data,
            "loss_l1": loss_l1.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_l1_sum = sum(log.get("loss_l1", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=9
        )
        metrics.log_scalar(
            "loss_l1_sum", loss_l1_sum / sample_size, sample_size, round=9
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_prediction_pdbbind", dataclass=GraphPredictionConfig)
class GraphPredictionLossPDBbind(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.loss_type = task.cfg.loss_type
        self.std_type = task.cfg.std_type
        self.readout_type = task.cfg.readout_type

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        logits = model(**sample["net_input"], modality = 'complex')[0]
        logits_padding = sample['net_input']['batched_data']['x'][:, :, 0].eq(0).unsqueeze(-1)

        if self.readout_type == 'cls':
            logits = logits[:, 0, :]
        elif self.readout_type == 'mean':
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).mean(dim=1)
        else:
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).sum(dim=1)

        targets = model.get_targets(sample, [logits])

        mean = sample['net_input']['batched_data']['mean']
        std = sample['net_input']['batched_data']['std']

        if self.std_type == 'std_labels':
            new_labels = (targets - mean) / std
            new_logits = logits
        elif self.std_type == 'std_logits':
            new_labels = targets
            new_logits = logits * std + mean
        else:
            new_labels = targets
            new_logits = logits

        if self.std_type == 'no_std':
            loss_l1 = nn.L1Loss(reduction="sum")(logits, targets)
        else:
            loss_l1 = nn.L1Loss(reduction="sum")(logits * std + mean, targets)

        if sample['net_input']['batched_data']['type'] == 'train':
            loss_func = nn.L1Loss if self.loss_type == 'L1' else nn.MSELoss
            loss = loss_func(reduction="sum")(new_logits, new_labels)
        else:
            loss = loss_l1

        logging_output = {
            "loss": loss.data,
            "loss_l1": loss_l1.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_l1_sum = sum(log.get("loss_l1", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=9
        )
        metrics.log_scalar(
            "loss_l1_sum", loss_l1_sum / sample_size, sample_size, round=9
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    

@register_criterion("graph_prediction_pdbbind_with_flag", dataclass=GraphPredictionConfig)
class GraphPredictionLossPDBbindWithFlag(GraphPredictionLossPDBbind):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(GraphPredictionConfig, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]['x'].shape[1]

        logits = model(**sample["net_input"], modality='complex', perturb=perturb)[0]
        logits_padding = sample['net_input']['batched_data']['x'][:, :, 0].eq(0).unsqueeze(-1)
        
        if self.readout_type == 'cls':
            logits = logits[:, 0, :]
        elif self.readout_type == 'mean':
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).mean(dim=1)
        else:
            logits = logits[:, 1:, :].masked_fill_(logits_padding, 0.0).sum(dim=1)

        targets = model.get_targets(sample, [logits])

        mean = sample['net_input']['batched_data']['mean']
        std = sample['net_input']['batched_data']['std']

        if self.std_type == 'std_labels':
            new_labels = (targets - mean) / std
            new_logits = logits
        elif self.std_type == 'std_logits':
            new_labels = targets
            new_logits = logits * std + mean
        else:
            new_labels = targets
            new_logits = logits

        if self.std_type == 'no_std':
            loss_l1 = nn.L1Loss(reduction="sum")(logits, targets)
        else:
            loss_l1 = nn.L1Loss(reduction="sum")(logits * std + mean, targets)

        if sample['net_input']['batched_data']['type'] == 'train':
            loss_func = nn.L1Loss if self.loss_type == 'L1' else nn.MSELoss
            loss = loss_func(reduction="sum")(new_logits, new_labels)
        else:
            loss = loss_l1

        logging_output = {
            "loss": loss.data,
            "loss_l1": loss_l1.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_l1_sum = sum(log.get("loss_l1", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=9
        )
        metrics.log_scalar(
            "loss_l1_sum", loss_l1_sum / sample_size, sample_size, round=9
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_finetune_l1_loss", dataclass=GraphFinetuneConfig)
class GraphPredictionL1Loss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"], modality='ligand')[0]
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])

        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_finetune_binary_logloss", dataclass=GraphFinetuneConfig)
class GraphPredictionBinaryLogLoss(FairseqCriterion):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"], modality='ligand')[0]
        logits = logits[:, 0, :]
        # targets = model.get_targets(sample, [logits])
        targets = sample["net_input"]["batched_data"]["y"]
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        logits_flatten = logits.reshape(-1)
        # targets_flatten = targets[: logits.size(0)].reshape(-1)
        targets_flatten = targets.reshape(-1)
        mask = ~torch.isnan(targets_flatten)

        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": torch.sum(mask.type(torch.int64)),
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": (preds == targets.view(logits.shape)).sum(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_finetune_binary_logloss_with_flag", dataclass=GraphFinetuneConfig)
class GraphPredictionBinaryLogLossWithFlag(FairseqCriterion):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]
        perturb = sample.get("perturb", None)

        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        logits = model(**sample["net_input"], modality='ligand', perturb=perturb)[0]
        logits = logits[:, 0, :]
        targets = model.get_targets(sample, [logits])
        preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        logits_flatten = logits.reshape(-1)
        targets_flatten = targets[: logits.size(0)].reshape(-1)
        mask = ~torch.isnan(targets_flatten)

        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": torch.sum(mask.type(torch.int64)),
            "nsentences": sample_size,
            "ntokens": natoms,
            "ncorrect": (preds == targets[:preds.size(0)]).sum(),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("graph_prediction_pcq", dataclass=GraphPredictionConfig)
class GraphPredictionLossPCQ(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPredictionConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        # self.loss_type = task.cfg.loss_type
        # self.std_type = task.cfg.std_type
        # self.readout_type = task.cfg.readout_type
        self.noise_scale = task.cfg.noise_scale

    def forward(self, model, sample, modality, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = len(sample['idx'])

        with torch.no_grad():
            natoms = sample['x'].shape[1]

        # add gaussian noise
        ori_pos = sample['pos']
        noise = torch.randn_like(ori_pos)
        noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        if modality == 'complex':
            # keep pocket fixed
            noise_mask[:, sample['max_len_ligand']:] = True
        noise = noise.masked_fill_(noise_mask, 0.0)
        sample['pos'] = ori_pos + noise * self.noise_scale

        token_mask = sample['x'][:, :, 0] == 121
        atom_token_no_padding = sample['x'][:, :, 0] != 0 if modality != 'complex' \
            else sample['x'][:, : sample['max_len_ligand'], 0] != 0

        with torch.no_grad():
            n_mask_node = token_mask.sum()
            total_atom_num = atom_token_no_padding.sum()
            #
            # sample['x'][token_mask] = sample['mask_node_label'] + 1

        model_output = model(sample, modality)
        logits, noise_output = model_output[0], model_output[1]
        logits = logits[:, 0, :].reshape(-1)
        targets = sample['y'].reshape(-1)
        loss_main = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        if noise_output is not None:
            # node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + noise_mask
            # node_output = node_output.masked_fill_(node_mask, 0.0)
            #
            # node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32),
            #                                                       noise.masked_fill_(node_mask, 0.0).to(
            #                                                           torch.float32)))
            # node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(
            #     torch.float16)
            #
            # tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
            # tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            # node_output_loss = (node_output_loss / tgt_count).sum() * 1

            node_mask = (noise_output == 0.0).all(
                dim=-1).all(dim=-1)[:, None, None] + noise_mask
            noise_output = noise_output.masked_fill_(node_mask, 0.0)
            noise_output_loss = F.mse_loss(noise_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32), reduction='none').sum(-1)
            noise_output_loss = noise_output_loss.masked_fill_(
                node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)

            tgt_count = (~node_mask).squeeze(-1).sum(dim=-
                                                     1).to(noise_output_loss)
            tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            noise_output_loss = (noise_output_loss / tgt_count).sum() * 1
        else:
            noise_output_loss = (noise - noise).sum()


        denoise_w = 1.0
        main_w = 1.0
        logging_output = {
            "noise_output_loss": denoise_w * noise_output_loss.data,
            "loss_l1": main_w * loss_main.data,
            "total_loss": denoise_w * noise_output_loss.data + main_w * loss_main.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "total_atom_num": total_atom_num  # sample["tokenizer_x"].shape[0]
        }
        return denoise_w * noise_output_loss + main_w * loss_main, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_l1_sum = sum(log.get("loss_l1", 0) for log in logging_outputs)
        noise_output_loss_sum = sum(log.get("noise_output_loss", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        total_atom_num = sum(log.get("total_atom_num", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss_l1_sum", loss_l1_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "noise_output_loss_sum", noise_output_loss_sum / total_atom_num, total_atom_num, round=6
        )
        metrics.log_scalar(
            "total_loss_sum", total_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True