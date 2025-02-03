from typing import Callable, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from .multihead_attention import MultiheadAttention
from .multiway_network import MultiwayWrapper, set_split_position
from .droppath import DropPath

class BiTEncoderLayer(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        with_complexffn: bool = False,
        is_mode: bool = True
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = FairseqDropout(
                dropout, module_name=self.__class__.__name__
            )

        # Initialize blocks
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            subln=sandwich_ln
        )

        self.sandwich_ln = sandwich_ln

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.ffn = MultiwayWrapper(FeedForwardNetwork(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            activation_fn = activation_fn,
            dropout = dropout, 
            activation_dropout = activation_dropout,
            module_name = self.__class__.__name__,
            subln=self.sandwich_ln
            ), multiway=is_mode)
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = MultiwayWrapper(LayerNorm(self.embedding_dim, export=export), multiway=is_mode)

        # self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        
        self.complex_ffn = None
        if with_complexffn:
            self.complex_ffn = FeedForwardNetwork(
                ffn_embedding_dim,
                self.embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                dropout = dropout, 
                activation_fn = activation_fn,
                activation_dropout = activation_dropout,
                module_name = self.__class__.__name__,
                subln=self.sandwich_ln
                )
            self.final_layer_norm_complex = LayerNorm(self.embedding_dim, export=export)
        
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        subln,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            subln=subln
        )
    
    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        multiway_split_position: int = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        if multiway_split_position is not None:
            self.apply(set_split_position(multiway_split_position))

        # x: T x B x C
        residual = x
        if self.sandwich_ln: 
            x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.sandwich_ln:
            x = self.self_attn_layer_norm(x)

        if self.complex_ffn is None:
            residual = x
            if self.sandwich_ln:
                x = self.final_layer_norm(x)
            x = self.ffn(x)
            x = self.dropout_module(x)
            x = residual + x
            if not self.sandwich_ln:
                x = self.final_layer_norm(x)

        else:
            residual = x
            if self.sandwich_ln:
                x = self.final_layer_norm_complex(x)
            x = self.complex_ffn(x)
            x = self.dropout_module(x)
            x = residual + x
            if not self.sandwich_ln:
                x = self.final_layer_norm_complex(x)

        return x, attn
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        ffn_dim = None,
        output_dim= None,
        q_noise = None, 
        qn_block_size = None,
        activation_fn = None,
        dropout = None,
        activation_dropout = None,
        module_name = None,
        layernorm_eps = None,
        subln=False
    ):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=module_name)
        self.dropout_module = FairseqDropout(dropout, module_name=module_name)
        self.fc1 = quant_noise(nn.Linear(output_dim, ffn_dim ), q_noise, qn_block_size)
        self.fc2 = quant_noise(nn.Linear(ffn_dim , output_dim), q_noise, qn_block_size)
        self.ffn_layernorm = LayerNorm(ffn_dim) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self,x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        # x = self.dropout_module(x)
        return x