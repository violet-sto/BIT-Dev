import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fairseq.modules import (
    FairseqDropout,
)

from fairseq import utils

from torch import Tensor
from typing import Callable, Optional


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


# class AtomFeature(nn.Module):
#     """
#     Compute atom features for each atom in the molecule.
#     """

#     def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, no_2d=False):
#         super(AtomFeature, self).__init__()
#         self.num_heads = num_heads
#         self.num_atoms = num_atoms
#         self.no_2d = no_2d

#         # 1 for graph token
#         self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
#         self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
#         self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

#         self.graph_token = nn.Embedding(1, hidden_dim)

#         self.apply(lambda module: init_params(module, n_layers=n_layers))

#     def forward(self, batched_data, num_dim=None, start_idx=None, end_idx=None, mask_2d=None, reverse_cls=False):
#         if start_idx is None and end_idx is not None:
#             # ligand of complex
#             x, in_degree, out_degree = batched_data['x'][:,:end_idx],batched_data['in_degree'][:,:end_idx], batched_data['out_degree'][:,:end_idx]
#         elif start_idx is not None and end_idx is None:
#             # pocket of complex
#             x, in_degree, out_degree = batched_data['x'][:,start_idx:,:num_dim],batched_data['in_degree'][:,start_idx:], batched_data['out_degree'][:,start_idx:]
#         elif start_idx is None and end_idx is None:
#             # unbound ligand or pocket
#             x, in_degree, out_degree = batched_data['x'],batched_data['in_degree'], batched_data['out_degree']
#         n_graph, n_node = x.size()[:2]

#         # node feauture + graph token
#         node_feature = self.atom_encoder(x).sum(dim=-2) # [n_graph, n_node, n_hidden]

#         if not self.no_2d:
#             degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
#             if mask_2d is not None:
#                 degree_feature = degree_feature * mask_2d[:, None, None]
#             node_feature = node_feature + degree_feature

#         graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
#         if reverse_cls:
#             graph_node_feature = torch.cat([node_feature, graph_token_feature], dim=1)
#         else:
#             graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

#         return graph_node_feature


class AtomFeature(nn.Module):
    """
    Compute atom features for each atom in the molecule.
    """

    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, no_2d=False):
        super(AtomFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_2d = no_2d

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(2, hidden_dim)
        self.type_feature = nn.Embedding(2, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, modality, mask_2d=None):
        x, in_degree, out_degree = batched_data['x'], batched_data['in_degree'], batched_data['out_degree']
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        if not self.no_2d:
            degree_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]
            node_feature = node_feature + degree_feature

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        if modality == 'ligand':
            multiway_split_position = -1
            graph_node_feature = torch.cat([graph_token_feature[:, :1], node_feature],
                                           dim=1)  # [n_graph, n_node+1, n_hidden]
            padding_mask = (graph_node_feature == 0.0).all(dim=-1, keepdim=True)
            graph_node_feature = graph_node_feature + self.type_feature.weight[0]
            graph_node_feature.masked_fill_(padding_mask, 0.0)

        elif modality == 'pocket':
            multiway_split_position = 0
            graph_node_feature = torch.cat([graph_token_feature[:, 1:], node_feature],
                                           dim=1)  # [n_graph, n_node+1, n_hidden]
            padding_mask = (graph_node_feature == 0.0).all(dim=-1, keepdim=True)
            graph_node_feature = graph_node_feature + self.type_feature.weight[1]
            graph_node_feature.masked_fill_(padding_mask, 0.0)

        elif modality == 'complex':
            multiway_split_position = batched_data['max_len_ligand'] + 1
            graph_node_feature = torch.cat([graph_token_feature[:, :1], node_feature, graph_token_feature[:, 1:]],
                                           dim=1)  # [n_graph, n_node+2, n_hidden]
            padding_mask = (graph_node_feature == 0.0).all(dim=-1, keepdim=True)
            graph_node_feature[:, :multiway_split_position] = graph_node_feature[:, :multiway_split_position] + \
                                                              self.type_feature.weight[0]
            graph_node_feature[:, multiway_split_position:] = graph_node_feature[:, multiway_split_position:] + \
                                                              self.type_feature.weight[1]
            graph_node_feature.masked_fill_(padding_mask, 0.0)

        return graph_node_feature, multiway_split_position


class ResidueFeature(nn.Module):
    """
    Compute residue features for each residue in the protein.
    """

    def __init__(self, num_heads, num_atoms, num_embeddings, hidden_dim, n_layers, no_2d=False):
        super(ResidueFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_2d = no_2d

        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.position_encoder = LearnedPositionalEmbedding(num_embeddings, hidden_dim, padding_idx=0)
        # self.position_encoder = SinusoidalPositionalEmbedding(hidden_dim, padding_idx=0)

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, start_idx=None, mask_2d=None):
        if start_idx is not None:
            # pocket of complex
            x = batched_data['x'][:, start_idx:, 0]
        else:
            # unbound pocket
            x = batched_data['x']
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x)  # [n_graph, n_node, n_hidden]

        if not self.no_2d:
            position_feature = self.position_encoder(x)
            if mask_2d is not None:
                position_feature = position_feature * mask_2d[:, None, None]
            node_feature = node_feature + position_feature

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb


class MoleculeAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, num_heads, num_atoms, num_edges, num_spatial, num_edge_dis, hidden_dim, edge_type,
                 multi_hop_max_dist, n_layers, no_2d=False):
        super(MoleculeAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.no_2d = no_2d
        self.n_layers = n_layers

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)

        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        # self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        # TODO:
        """
        """
        self.graph_token_virtual_distance1 = nn.Embedding(1, num_heads)
        self.graph_token_virtual_distance2 = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=self.n_layers))

    def reset_parameters(self):
        self.apply(lambda module: init_params(module, n_layers=self.n_layers))

    def forward(self, batched_data, mask_2d=None, modality="base"):
        attn_bias, spatial_pos, x = batched_data['attn_bias'], batched_data['spatial_pos'], batched_data['x']
        edge_input, attn_edge_type = batched_data['edge_input'], batched_data['attn_edge_type']

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if not self.no_2d:
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            if mask_2d is not None:
                spatial_pos_bias = spatial_pos_bias * mask_2d[:, None, None, None]
            # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
            #                                             :, 1:, 1:] + spatial_pos_bias
            # TODO:
            """
            """
            if modality == "complex":
                graph_attn_bias[:, :, 1:-1, 1:-1] = graph_attn_bias[:,
                                                    :, 1:-1, 1:-1] + spatial_pos_bias
            else:
                graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        # t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        # graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        # graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        # TODO:
        """
        """
        if modality == "complex":
            t1 = self.graph_token_virtual_distance1.weight.view(1, self.num_heads, 1)
            t2 = self.graph_token_virtual_distance2.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:-1, 0] = graph_attn_bias[:, :, 1:-1, 0] + t1
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t1
            graph_attn_bias[:, :, 1:-1, -1] = graph_attn_bias[:, :, 1:-1:, -1] + t2
            graph_attn_bias[:, :, -1, :] = graph_attn_bias[:, :, -1, :] + t2
        elif modality == "ligand":
            t1 = self.graph_token_virtual_distance1.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t1
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t1
        elif modality == "pocket":
            t2 = self.graph_token_virtual_distance2.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t2
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t2

        if not self.no_2d:

            # edge feature
            if self.edge_type == 'multi_hop':
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
                if self.multi_hop_max_dist > 0:
                    spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                    edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
                # [n_graph, n_node, n_node, max_dist, n_head]
                edge_input = self.edge_encoder(edge_input).mean(-2)
                max_dist = edge_input.size(-2)
                edge_input_flat = edge_input.permute(
                    3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
                edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads)[:max_dist, :, :])
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
                edge_input = (edge_input.sum(-2) /
                              (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)

            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = self.edge_encoder(
                    attn_edge_type).mean(-2).permute(0, 3, 1, 2)

            if mask_2d is not None:
                edge_input = edge_input * mask_2d[:, None, None, None]
            # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
            #                                                 :, 1:, 1:] + edge_input
            # TODO：
            """
            """
            if modality == "complex":
                graph_attn_bias[:, :, 1:-1, 1:-1] = graph_attn_bias[:,
                                                    :, 1:-1, 1:-1] + edge_input
            else:
                graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                :, 1:, 1:] + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class Molecule3DBias(nn.Module):
    """
        Compute 3D attention bias according to the position information for each head.
        """

    def __init__(self, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False):
        super(Molecule3DBias, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.gbf.reset_parameters()
        self.gbf_proj.reset_parameters()

    def forward(self, batched_data):

        pos, x, node_type_edge = batched_data['pos'], batched_data['x'], batched_data[
            'node_type_edge']  # pos shape: [n_graphs, n_nodes, 3]
        # pos.requires_grad_(True)

        padding_mask = x.eq(0).all(dim=-1)
        n_graph, n_node, _ = pos.shape
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        edge_feature = self.gbf(dist, torch.zeros_like(dist).long() if node_type_edge is None else node_type_edge.long())
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        # keep pe the same for ligand/pocket and complex
        try:
            max_len_ligand = batched_data['max_len_ligand']
            sum_edge_features = torch.cat(
                (edge_feature[:, :max_len_ligand, :max_len_ligand].sum(dim=-2),
                 edge_feature[:, max_len_ligand:, max_len_ligand:].sum(dim=-2)),
                dim=-2)
        except:
            sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        return graph_attn_bias, merge_edge_features, delta_pos


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def reset_parameters(self):
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class AtomTaskHead(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.dropout_module = FairseqDropout(
            0.1, module_name=self.__class__.__name__
        )

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(
            self,
            query: Tensor,
            attn_bias: Tensor,
            delta_pos: Tensor,
    ) -> Tensor:
        query = query.contiguous().transpose(0, 1)
        bsz, n_node, _ = query.size()
        q = (
                self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
                * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs_float = utils.softmax(
            attn.view(-1, n_node, n_node) + attn_bias.contiguous().view(-1, n_node, n_node), dim=-1, onnx_trace=False)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(bsz, self.num_heads, n_node, n_node)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force