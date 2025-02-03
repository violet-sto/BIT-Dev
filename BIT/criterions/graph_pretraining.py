from dataclasses import dataclass
import math
from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import os

from ..models.tokenizer import GNN, VectorQuantizer

@dataclass
class GraphPreTrainingConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("graph_pretraining", dataclass=GraphPreTrainingConfig)
class GraphPreTrainingLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, cfg: GraphPreTrainingConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.noise_scale = task.cfg.noise_scale
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # gnn、tokenizer
        # self.gnn = GNN(5, 300)
        # self.tokenizer = VectorQuantizer(300, 512, 0.25)
        # self.gnn.from_pretrained(f"../logs/tokenizer/checkpoints30_vqencoder_pcqm_512.pth")
        # self.tokenizer.from_pretrained(f"../logs/tokenizer/checkpoints30_vqquantizer_pcqm_512.pth")
        # self.gnn.eval()
        # self.tokenizer.eval()


    def forward(self, model, sample, modality, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = len(sample['idx'])


        # add gaussian noise
        ori_pos = sample['pos']
        noise = torch.randn_like(ori_pos)
        noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        if modality == 'complex':
            # keep pocket fixed
            noise_mask[:, sample['max_len_ligand']:] = True
        noise = noise.masked_fill_(noise_mask, 0.0)
        sample['pos'] = ori_pos + noise * self.noise_scale

        # MAM, group VQ-VAE as Tokenizer
        # with torch.no_grad():
        #     e = self.gnn(sample["tokenizer_x"], sample["tokenizer_edge_index"], sample["tokenizer_edge_attr"])
        #     encoding_indices = self.tokenizer.get_code_indices(sample["tokenizer_x"][:,:2], e)
        #     sample['mask_node_label'] = encoding_indices[sample["tokenizer_mask_atom_indices"]].unsqueeze(1)

        token_mask = sample['x'][:, :, 0] == 121
        model_output = model(sample, modality)
        logits, noise_output = model_output[0], model_output[1]
        if modality == 'complex':
            # double cls token
            logits = logits[:, 1: -1, :]
        else:
            logits = logits[:, 1:, :]

        if noise_output is not None:
            node_mask = (noise_output == 0.0).all(
                dim=-1).all(dim=-1)[:, None, None] + noise_mask
            noise_output = noise_output.masked_fill_(node_mask, 0.0)

            # noise_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(noise_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
            noise_output_loss = F.mse_loss(noise_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32), reduction='none').mean(-1)
            noise_output_loss = noise_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)

            tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(noise_output_loss)
            tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            noise_output_loss = (noise_output_loss / tgt_count).sum() * 1
        else:
            noise_output_loss = (noise - noise).sum()

        element_output_loss = sample_size * self.criterion(
            logits[token_mask], sample['mask_node_label'][:, 0])

        with torch.no_grad():
            natoms = sample['x'].shape[1]
            mask_node_num = token_mask.sum()
            denoise_mol_num = sample_size - (noise_output == 0.0).all(dim=-1).all(dim=-1).sum() \
                if noise_output is not None else sample_size
            node_correct_num = self.compute_correct(
                logits[token_mask], sample['mask_node_label'][:, 0])

        denoise_w = 1.0
        mask_w = 0.2
        logging_output = {
            "noise_output_loss": noise_output_loss.data,
            "element_output_loss": element_output_loss.data,
            "total_loss": denoise_w * noise_output_loss.data + mask_w * element_output_loss.data,
            "node_correct": node_correct_num,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "mask_node_num": mask_node_num,
            "denoise_mol_num": denoise_mol_num   # tgt_count.sum().data  # sample["tokenizer_x"].shape[0]
        }
        return denoise_w * noise_output_loss + mask_w * element_output_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        noise_output_loss_sum = sum(
            log.get("noise_output_loss", 0) for log in logging_outputs)
        element_output_loss_sum = sum(
            log.get("element_output_loss", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("total_loss", 0)
                             for log in logging_outputs)
        node_cor_sum = sum(log.get("node_correct", 0) for log in logging_outputs)
        mask_node_num = sum(log.get("mask_node_num", 0) for log in logging_outputs)
        denoise_mol_num = sum(log.get("denoise_mol_num", 0) for log in logging_outputs)

        ligand_loss_sum = sum(log.get("ligand:total_loss", 0) for log in logging_outputs)
        pocket_loss_sum = sum(log.get("pocket:total_loss", 0) for log in logging_outputs)
        complex_loss_sum = sum(log.get("complex:total_loss", 0) for log in logging_outputs)
        ligand_sample_size = sum(log.get("ligand:sample_size", 0) for log in logging_outputs)
        pocket_sample_size = sum(log.get("pocket:sample_size", 0) for log in logging_outputs)
        complex_sample_size = sum(log.get("complex:sample_size", 0) for log in logging_outputs)

        ligand_noise_output_loss_sum = sum(log.get("ligand:noise_output_loss", 0) for log in logging_outputs)
        pocket_noise_output_loss_sum = sum(log.get("pocket:noise_output_loss", 0) for log in logging_outputs)
        complex_noise_output_loss_sum = sum(log.get("complex:noise_output_loss", 0) for log in logging_outputs)

        ligand_node_cor_sum = sum(log.get("ligand:node_correct", 0) for log in logging_outputs)
        pocket_node_cor_sum = sum(log.get("pocket:node_correct", 0) for log in logging_outputs)
        complex_node_cor_sum = sum(log.get("complex:node_correct", 0) for log in logging_outputs)
        ligand_mask_node_num = sum(log.get("ligand:mask_node_num", 0) for log in logging_outputs)
        pocket_mask_node_num = sum(log.get("pocket:mask_node_num", 0) for log in logging_outputs)
        complex_mask_node_num = sum(log.get("complex:mask_node_num", 0) for log in logging_outputs)
        ligand_denoise_mol_num = sum(log.get("ligand:denoise_mol_num", 0) for log in logging_outputs)
        pocket_denoise_mol_num = sum(log.get("pocket:denoise_mol_num", 0) for log in logging_outputs)
        complex_denoise_mol_num = sum(log.get("complex:denoise_mol_num", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "total_loss", total_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "noise_output_loss", noise_output_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "element_output_loss", element_output_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "acc_node", node_cor_sum / mask_node_num, mask_node_num, round=6
        )
        metrics.log_scalar(
            "noise_loss_per_mol", noise_output_loss_sum / denoise_mol_num, denoise_mol_num, round=6
        )

        # domain level
        metrics.log_scalar(
            "ligand:total_loss", ligand_loss_sum / (ligand_sample_size + 1e-7), ligand_sample_size, round=6
        )
        metrics.log_scalar(
            "pocket:total_loss", pocket_loss_sum / (pocket_sample_size + 1e-7), pocket_sample_size, round=6
        )
        metrics.log_scalar(
            "complex:total_loss", complex_loss_sum / (complex_sample_size + 1e-7), complex_sample_size, round=6
        )

        metrics.log_scalar(
            "ligand:acc_node", ligand_node_cor_sum / (ligand_mask_node_num + 1e-7), ligand_mask_node_num, round=6
        )
        metrics.log_scalar(
            "pocket:acc_node", pocket_node_cor_sum / (pocket_mask_node_num + 1e-7), pocket_mask_node_num, round=6
        )
        metrics.log_scalar(
            "complex:acc_node", complex_node_cor_sum / (complex_mask_node_num + 1e-7), complex_mask_node_num, round=6
        )

        metrics.log_scalar(
            "ligand:noise_loss_per_mol", ligand_noise_output_loss_sum / (ligand_denoise_mol_num + 1e-7), ligand_denoise_mol_num, round=6
        )
        metrics.log_scalar(
            "pocket:noise_loss_per_mol", pocket_noise_output_loss_sum / (pocket_denoise_mol_num + 1e-7), pocket_denoise_mol_num, round=6
        )
        metrics.log_scalar(
            "complex:noise_loss_per_mol", complex_noise_output_loss_sum / (complex_denoise_mol_num + 1e-7), complex_denoise_mol_num, round=6
        )



    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    @staticmethod
    def compute_accuracy(pred, target):
        return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)

    @staticmethod
    def compute_correct(pred, target):
        return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())