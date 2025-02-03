# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch_geometric.data import Data, Batch
# from torchdrug.data.dataloader import graph_collate

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, exclude_keys=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()
        exclude_keys = set(exclude_keys or [])

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            if key in exclude_keys:
                continue
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_1d_unsqueeze_complex(x, len_ligand, max_node_num):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    xstart = max_node_num[0]-len_ligand
    padlen = sum(max_node_num)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[xstart:xstart+xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze_complex(x, len_ligand, max_node_num):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    xstart = max_node_num[0]-len_ligand
    padlen = sum(max_node_num)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[xstart:xstart+xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze_complex(x, len_ligand, max_node_num):
    xlen = x.size(0)
    padlen = sum(max_node_num)+1
    xstart = max_node_num[0]-len_ligand
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[xstart:xstart+xlen, xstart:xstart+xlen] = x
        new_x[:xstart, xstart:xstart+xlen] = 0
        new_x[xstart+xlen:, xstart:xstart+xlen] = 0
        x = new_x
    return x.unsqueeze(0)

# def pad_attn_bias_unsqueeze_complex_cls(x, len_ligand, max_node_num):
#     xlen = x.size(0)
#     padlen = sum(max_node_num)+2
#     xstart = max_node_num[0]-len_ligand
#     if xlen < padlen:
#         new_x = x.new_zeros(
#             [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
#         new_x[xstart:xstart+xlen, xstart:xstart+xlen] = x
#         new_x[:xstart, xstart:xstart+xlen] = 0
#         new_x[xstart+xlen:, xstart:xstart+xlen] = 0
#         x = new_x
#     return x.unsqueeze(0)

def pad_attn_bias_unsqueeze_complex_cls(x, len_ligand, max_node_num):
    xlen = x.size(0)
    padlen = sum(max_node_num) + 2
    xstart = max_node_num[0] - len_ligand
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[xstart:xstart + xlen, xstart:xstart + xlen] = x
        new_x[:xstart, xstart:xstart + xlen] = 0
        new_x[xstart + xlen:, xstart:xstart + xlen] = 0

        new_x[[0, xstart]] = new_x[[xstart, 0]]
        new_x[:, [0, xstart]] = new_x[:, [xstart, 0]]
        new_x[[-1, xstart + xlen - 1]] = new_x[[xstart + xlen - 1, -1]]
        new_x[:, [-1, xstart + xlen - 1]] = new_x[:, [xstart + xlen - 1, -1]]

        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze_complex(x, len_ligand, max_node_num):
    xlen = x.size(0)
    padlen = sum(max_node_num)
    xstart = max_node_num[0]-len_ligand
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[xstart:xstart+xlen, xstart:xstart+xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze_complex(x, len_ligand, max_node_num):
    x = x + 1
    xlen = x.size(0)
    xstart = max_node_num[0]-len_ligand
    padlen = sum(max_node_num)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[xstart:xstart+xlen, xstart:xstart+xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze_complex(x, len_ligand, max_node_num, padlen2):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    xstart = max_node_num[0]-len_ligand
    padlen1 = sum(max_node_num)
    if xlen1 < padlen1 or xlen2 < padlen1 or xlen3 < padlen2:
        new_x = x.new_zeros([padlen1, padlen1, padlen2, xlen4], dtype=x.dtype)
        new_x[xstart:xstart+xlen1, xstart:xstart+xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_unsqueeze_complex(x, len_ligand, max_node_num):
    xlen, xdim = x.size()
    xstart = max_node_num[0]-len_ligand
    padlen = sum(max_node_num)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[xstart:xstart+xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:,
                                                       :, :multi_hop_max_dist, :], item.y,
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx]
                                 >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                             for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
    )


def collator_3d(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,

        pos=pos,
        node_type_edge=node_type_edge,
    )

def collator_pretrain(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    raw_items = [item.raw_item for item in items]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.pos,
              item.mask_node_label
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses\
        , in_degrees, out_degrees, xs, edge_inputs, poses, mask_node_labels = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                             for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0] # item.x
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    tokenizer_batch = BatchMasking.from_data_list(raw_items, exclude_keys=['idx', 'num_nodes', 'residues'])
    # masked_atom_indices = torch.zeros(pyg_graphs.x.size(0), dtype=torch.bool)
    # masked_atom_indices[pyg_graphs.masked_atom_indices] = True

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,

        pos=pos,
        node_type_edge=node_type_edge,
        mask_node_label=torch.cat(mask_node_labels),

        tokenizer_x = tokenizer_batch.x[:,:2],
        tokenizer_edge_attr=tokenizer_batch.edge_attr,
        tokenizer_edge_index=tokenizer_batch.edge_index,
        tokenizer_mask_atom_indices=tokenizer_batch.masked_atom_indices
    )

def collator_2d_moleculenet(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:,
                                                       :, :multi_hop_max_dist, :], item.pos, item.y
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, poses, ys= zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx]
                                 >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                             for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(
            node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(
            node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        y=y,
        edge_input=edge_input,
        pos=pos,
        node_type_edge=node_type_edge
    )

def collator_3d_qm9(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos,
              item.train_mean, item.train_std, item.type

              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, means, stds, types = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,

        pos=pos,
        node_type_edge=node_type_edge,

        mean=means[0],
        std=stds[0],
        type=types[0],
    )

def collator_protein(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    
    
    
    # items = [(item.idx, item['graph'].residue_type,   
    #           ) for item in items]

    pass

def collator_complex(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    if len(items) == 0:
        return dict()
    
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:,
                                                       :, :multi_hop_max_dist, :], item.pos,
              item.len_ligand, item.len_pocket, item.mask_node_label
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, poses, len_ligands, len_pockets, mask_node_labels = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx]
                                 >= spatial_pos_max] = float('-inf')
    # max_node_num = max(i.size(0) for i in xs)
    max_node_num = (max(len_ligands), max(len_pockets))

    max_dist = max(i.size(-2)
                   for i in edge_inputs)  

    x = torch.cat([pad_2d_unsqueeze_complex(i, len_ligands[idx],
                  max_node_num) for idx, i in enumerate(xs)])
    edge_input = torch.cat([pad_3d_unsqueeze_complex(i, len_ligands[idx],
                                                     max_node_num, max_dist) for idx, i in enumerate(edge_inputs)])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze_complex(i, len_ligands[idx],
                                                           max_node_num) for idx, i in enumerate(attn_biases)])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze_complex(i, len_ligands[idx],
                                                                max_node_num) for idx, i in enumerate(attn_edge_types)])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze_complex(i, len_ligands[idx],
                                                               max_node_num) for idx, i in enumerate(spatial_poses)])
    in_degree = torch.cat([pad_1d_unsqueeze_complex(i, len_ligands[idx],
                                                    max_node_num) for idx, i in enumerate(in_degrees)])

    pos = torch.cat([pad_pos_unsqueeze_complex(i, len_ligands[idx],
                                               max_node_num) for idx, i in enumerate(poses)])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze_complex(
            node_atom_i, len_ligands[idx], max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze_complex(
            node_atom_j, len_ligands[idx], max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        max_len_ligand=max(len_ligands),
        max_len_pocket=max(len_pockets),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,

        pos=pos,
        node_type_edge=node_type_edge,
        mask_node_label=torch.cat(mask_node_labels)
    )

def collator_complex_cls(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    if len(items) == 0:
        print("=====DEBUG=====")
        # print(w[0].x.size(0))
        return dict()
    
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:,
                                                       :, :multi_hop_max_dist, :], item.pos,
              item.len_ligand, item.len_pocket, item.mask_node_label
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, poses, len_ligands, len_pockets, mask_node_labels = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:-1, 1:-1][spatial_poses[idx]
                                 >= spatial_pos_max] = float('-inf')
    # max_node_num = max(i.size(0) for i in xs)
    max_node_num = (max(len_ligands), max(len_pockets))

    max_dist = max(i.size(-2)
                   for i in edge_inputs)  

    x = torch.cat([pad_2d_unsqueeze_complex(i, len_ligands[idx],
                  max_node_num) for idx, i in enumerate(xs)])
    edge_input = torch.cat([pad_3d_unsqueeze_complex(i, len_ligands[idx],
                                                     max_node_num, max_dist) for idx, i in enumerate(edge_inputs)])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze_complex_cls(i, len_ligands[idx],
                                                           max_node_num) for idx, i in enumerate(attn_biases)])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze_complex(i, len_ligands[idx],
                                                                max_node_num) for idx, i in enumerate(attn_edge_types)])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze_complex(i, len_ligands[idx],
                                                               max_node_num) for idx, i in enumerate(spatial_poses)])
    in_degree = torch.cat([pad_1d_unsqueeze_complex(i, len_ligands[idx],
                                                    max_node_num) for idx, i in enumerate(in_degrees)])

    pos = torch.cat([pad_pos_unsqueeze_complex(i, len_ligands[idx],
                                               max_node_num) for idx, i in enumerate(poses)])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze_complex(
            node_atom_i, len_ligands[idx], max_node_num).unsqueeze(-1)
        # node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = (node_atom_type).unsqueeze(0).repeat(n_nodes, 1) # Distinguish between ligand and pocket
        node_atom_j = pad_spatial_pos_unsqueeze_complex(
            node_atom_j, len_ligands[idx], max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        max_len_ligand=max(len_ligands),
        max_len_pocket=max(len_pockets),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,

        pos=pos,
        node_type_edge=node_type_edge,
        mask_node_label=torch.cat(mask_node_labels)
    )


def collator_complex_pdbbind(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    if len(items) == 0:
        return dict()
    
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:,
                                                       :, :multi_hop_max_dist, :], item.y, item.pos,
              item.len_ligand, item.len_pocket, item.train_mean, item.train_std, item.type
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, len_ligands, len_pockets, means, stds, types = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx]
                                 >= spatial_pos_max] = float('-inf')
    # max_node_num = max(i.size(0) for i in xs)
    max_node_num = (max(len_ligands), max(len_pockets))

    max_dist = max(i.size(-2)
                   for i in edge_inputs)  
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze_complex(i, len_ligands[idx],
                  max_node_num) for idx, i in enumerate(xs)])
    edge_input = torch.cat([pad_3d_unsqueeze_complex(i, len_ligands[idx],
                                                     max_node_num, max_dist) for idx, i in enumerate(edge_inputs)])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze_complex(i, len_ligands[idx],
                                                           max_node_num) for idx, i in enumerate(attn_biases)])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze_complex(i, len_ligands[idx],
                                                                max_node_num) for idx, i in enumerate(attn_edge_types)])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze_complex(i, len_ligands[idx],
                                                               max_node_num) for idx, i in enumerate(spatial_poses)])
    in_degree = torch.cat([pad_1d_unsqueeze_complex(i, len_ligands[idx],
                                                    max_node_num) for idx, i in enumerate(in_degrees)])

    pos = torch.cat([pad_pos_unsqueeze_complex(i, len_ligands[idx],
                                               max_node_num) for idx, i in enumerate(poses)])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze_complex(
            node_atom_i, len_ligands[idx], max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze_complex(
            node_atom_j, len_ligands[idx], max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        max_len_ligand=max(len_ligands),
        max_len_pocket=max(len_pockets),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,

        pos=pos,
        node_type_edge=node_type_edge,

        mean=means[0],
        std=stds[0],
        type=types[0],
    )

def collator_complex_dude(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    anchor_dict = collator_3d([item[0] for item in items])
    ligands_dict = collator_3d(sum([item[1] for item in items],[]))

    return dict(
        anchor=anchor_dict,
        ligands=ligands_dict
    )