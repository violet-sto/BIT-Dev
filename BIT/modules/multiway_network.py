# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# From torchscale/component

import copy

import torch
import torch.nn as nn


def MultiwayWrapper(module, multiway=True, dim=0):
    if multiway:
        return MultiwayNetwork(module, dim=dim)
    return module

def MultiwayBiasWrapper(module, multiway=True, format="2D"):
    if multiway:
        return MultiwayBiasNetwork(module, format=format)
    return module

def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)

class MultiwayBiasNetwork(nn.Module):
    def __init__(self, module, format="2D"):
        super().__init__()
        self.format = format
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        if self.format == "2D":
            y1, y2 = self.A(x, **kwargs), self.B(x, **kwargs)
            n = y1.shape[2]
            mask1, mask2 = torch.zeros(n, n).to(y1.device), torch.zeros(n, n).to(y2.device)
            mask1, mask2 = torch.triu(mask1.fill_(1.0)), torch.tril(mask2.fill_(1.0), -1)
            y = torch.cat([y1[:, :, :self.split_position, :], y2[:, :, self.split_position:, :]], dim=2)

            # return y1 * mask1[None, None, :, :] + y2 * mask2[None, None, :, :]
            return y
            # return (y1 + y2) / 2
        elif self.format == "3D":
            y1, y2 = self.A(x, **kwargs), self.B(x, **kwargs)
            n = y1[0].shape[2]
            mask1, mask2 = torch.zeros(n, n).to(y1[0].device), torch.zeros(n, n).to(y2[0].device)
            mask1, mask2 = torch.triu(mask1.fill_(1.0)), torch.tril(mask2.fill_(1.0), -1)
            # graph_attn_bias = y1[0] * mask1[None, None, :, :] + y1[0] * mask2[None, None, :, :]
            graph_attn_bias_upper = torch.cat([y1[0][:, :, :self.split_position, :self.split_position], y2[0][:, :, :self.split_position, self.split_position:]], dim=3)
            graph_attn_bias_bot = torch.cat([y1[0][:, :, self.split_position:, :self.split_position], y2[0][:, :, self.split_position:, self.split_position:]], dim=3)
            graph_attn_bias = torch.cat([graph_attn_bias_upper, graph_attn_bias_bot],
                                        dim=2)


            # graph_attn_bias = torch.cat([y1[0][:, :, :self.split_position, :], y2[0][:, :, self.split_position:, :]],
            #                             dim=2)
            # graph_attn_bias = (y1[0] + y2[0]) / 2
            merge_edge_features = (y1[1] + y2[1]) / 2  # mean
            delta_pos = y1[2]  # y2[2] is also feasible
            return graph_attn_bias, merge_edge_features, delta_pos
        else:
            raise TypeError(self.format)

class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1