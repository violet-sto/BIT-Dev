import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from ..modules import (
    init_params,
    BiTEncoder,
    BiTEncoderPDBbind
)

logger = logging.getLogger(__name__)

@register_model("BIT")
class BiTModel(FairseqEncoderModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, "apply_init", False):
            self.apply(init_params)
        self.encoder_embed_dim = args.encoder_embed_dim


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--remove-head", action='store_true', help="remove pre-trained prediction head"
        )
        parser.add_argument(
            "--load-pdbbind", action='store_true', help="load pdbbind model"
        )
        parser.add_argument(
            "--mode-prob", type=str, default="0.2,0.2,0.6", help="probability of {2D+3D, 2D, 3D} mode for joint training"
        )
        parser.add_argument(
            "--mode-ffn", action='store_true', help="multiway of domain ffn"
        )
        parser.add_argument(
            "--mode-2d", action='store_true', help="multiway of 2d bias"
        )
        parser.add_argument(
            "--mode-3d", action='store_true', help="multiway of 3d bias"
        )
        parser.add_argument(
            "--add-3d", action='store_true', help="add 3D attention bias"
        )
        parser.add_argument(
            "--no-2d", action='store_true', help="remove 2D encodings"
        )
        parser.add_argument(
            "--num-3d-bias-kernel", type=int, default=128, metavar="D", help="number of kernel in 3D attention bias"
        )
        parser.add_argument(
            "--droppath-prob", type=float, metavar="D", help="stochastic path probability", default=0.0
        )

        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " prediction head weights",
        )
        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--complexffn-start-layer",
            type=int,
            help='layer index start complexffn '
        )
        
        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embeddings",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--num-segment", type=int, metavar="N", help="num segment in the input"
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        # Arguments related to sentence level prediction
        parser.add_argument(
            "--sentence-class-num",
            type=int,
            metavar="N",
            help="number of classes for sentence task",
        )
        parser.add_argument(
            "--sent-loss",
            action="store_true",
            help="if set," " calculate sentence level predictions",
        )

        # Arguments related to parameter initialization
        parser.add_argument(
            "--apply-init",
            action="store_true",
            help="use custom param initialization for Transformer-M",
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="Which activation function to use for pooler layer.",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )

    def forward(self, batched_data, modality, **kwargs):
        return self.encoder(batched_data, modality, **kwargs)

    def max_positions(self):
        return self.encoder.max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            try:
                args.max_positions = args.tokens_per_sample
            except:
                args.max_positions = args.max_nodes

        logger.info(args)

        if args.load_pdbbind:
            encoder = BiTPDBbind(args)
        else:
            encoder = BiT(args)

        return cls(args, encoder)


class BiT(FairseqEncoder):

    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_positions = args.max_positions

        self.molecule_encoder = BiTEncoder(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            complexffn_start_layer_index=args.complexffn_start_layer,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_init=args.apply_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            sandwich_ln=args.sandwich_ln,
            droppath_prob=args.droppath_prob,
            add_3d=args.add_3d,
            num_3d_bias_kernel=args.num_3d_bias_kernel,
            no_2d=args.no_2d,
            mode_prob=args.mode_prob,
            mode_ffn=args.mode_ffn,
            mode_2d=args.mode_2d,
            mode_3d=args.mode_3d
        )
        print("debug1,", args.mode_2d,  args.mode_3d)

        self.embed_out = None
        self.proj_out = None

        # remove_head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        if self.load_softmax:
            #TODO weather tie embeding?
            # self.mmm_head = RobertaLMHead(args.encoder_embed_dim, 119+2, args.activation_fn)
            # self.mpm_head = RobertaLMHead(args.encoder_embed_dim, 119+2, args.activation_fn)
            self.mmm_head = nn.Linear(args.encoder_embed_dim, 121)
            self.mpm_head = nn.Linear(args.encoder_embed_dim, 121)
            # implemented in Beit3
            # torch.nn.init.normal_(
            #     self.mmm_head.weight, mean=0, std=args.encoder_embed_dim**-0.5
            # )

        else:
            self.proj_out = ClassificationHead(
                    args.encoder_embed_dim, args.encoder_embed_dim, args.num_classes, args.activation_fn
                )

    def forward(self, batched_data, modality, perturb=None, segment_labels=None, masked_tokens=None, **unused):

        inner_states, noise_output = self.molecule_encoder(
            batched_data,
            modality=modality,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        if self.load_softmax:
            if modality in ['ligand', 'complex']:
                x = self.mmm_head(x)
            elif modality in ['pocket']:
                x = self.mpm_head(x)
        else:
            x = self.proj_out(x)
            
        return x, noise_output, {
            "inner_states": inner_states,
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        tmp_dict = {}
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k
                    or "sentence_projection_layer.weight" in k
                    or "lm_output_learned_bias" in k
                    or "node_proc" in k
                    or "proj_out.ln" in k
                    or "mmm_head" in k
                    or "mpm_head" in k
                ):
                    print("Removing", k, "(because load_softmax is False)")
                    tmp_dict[k] = state_dict[k]
                    del state_dict[k]

            may_missing_keys = [
                'encoder.proj_out.dense.weight',
                'encoder.proj_out.dense.bias',
                'encoder.proj_out.out_proj.weight',
                'encoder.proj_out.out_proj.bias',
                # 'encoder.lm_head_transform_weight.weight',
                # 'encoder.lm_head_transform_weight.bias',
                # 'encoder.layer_norm.weight',
                # 'encoder.layer_norm.bias',
            ]

            named_parameters = {
                "encoder." + k: v
                for k, v in self.named_parameters()
            }

            for k in may_missing_keys:
                if k not in state_dict.keys():
                    state_dict[k] = named_parameters[k].data
                    print("Copying", k, "(from model initialization)")
        return state_dict


class BiTPDBbind(FairseqEncoder):

    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_positions = args.max_positions

        self.molecule_encoder = BiTEncoderPDBbind(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            complexffn_start_layer_index=args.complexffn_start_layer,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_init=args.apply_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            sandwich_ln=args.sandwich_ln,
            droppath_prob=args.droppath_prob,
            add_3d=args.add_3d,
            num_3d_bias_kernel=args.num_3d_bias_kernel,
            no_2d=args.no_2d,
            mode_prob=args.mode_prob,
            mode_ffn=args.mode_ffn,
            mode_2d=args.mode_2d,
            mode_3d=args.mode_3d
        )

        self.embed_out = None
        self.proj_out = None

        # remove_head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)
        print("load_softmax", self.load_softmax)

        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(119+2))
            self.embed_out = nn.Linear(
                args.encoder_embed_dim, 119+2, bias=False
            )
        else:
            self.proj_out = ClassificationHead(
                    args.encoder_embed_dim, args.encoder_embed_dim, args.num_classes, args.activation_fn,
                    args.pooler_dropout
                )

    def forward(self, batched_data, modality, perturb=None, segment_labels=None, masked_tokens=None, **unused):

        inner_states, noise_output = self.molecule_encoder(
            batched_data,
            modality=modality,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        if self.load_softmax:
            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
            x = self.embed_out(x)
            x = x + self.lm_output_learned_bias
        else:
            x = self.proj_out(x)
            
        return x, noise_output, {
            "inner_states": inner_states,
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        tmp_dict = {}
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k
                    or "sentence_projection_layer.weight" in k
                    or "lm_output_learned_bias" in k
                    or "node_proc" in k
                    or "proj_out.ln" in k
                    or "mmm_head" in k
                    or "mpm_head" in k
                ):
                    print("Removing", k, "(because load_softmax is False)")
                    tmp_dict[k] = state_dict[k]
                    del state_dict[k]

            may_missing_keys = [
                'encoder.proj_out.dense.weight',
                'encoder.proj_out.dense.bias',
                'encoder.proj_out.out_proj.weight',
                'encoder.proj_out.out_proj.bias',
                'encoder.lm_head_transform_weight.weight',
                'encoder.lm_head_transform_weight.bias',
                'encoder.layer_norm.weight',
                'encoder.layer_norm.bias', ]

            named_parameters = {
                "encoder." + k: v
                for k, v in self.named_parameters()
            }

            for k in may_missing_keys:
                if k not in state_dict.keys():
                    state_dict[k] = named_parameters[k].data
                    print("Copying", k, "(from model initialization)")
        return state_dict
    

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class ClassificationHead(nn.Module):
    """Head for classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("BIT", "bio-interaction-transformer")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.num_segment = getattr(args, "num_segment", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_init = getattr(args, "apply_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    args.mode_ffn = getattr(args, "mode_ffn", False)
    args.mode_2d = getattr(args, "mode_2d", False)
    args.mode_3d = getattr(args, "mode_3d", False)


@register_model_architecture("BIT", "bio-interaction-transformer_base")
def bert_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_init = getattr(args, "apply_init", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    args.add_3d = getattr(args, "add_3d", False)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.no_2d = getattr(args, "no_2d", False)
    args.mode_prob = getattr(args, "mode_prob", "0.2,0.2,0.6")

    args.load_pdbbind = getattr(args, "load_pdbbind", False)
    args.load_dude = getattr(args, "load_dude", False)
    args.remove_head = getattr(args, "remove_head", False)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.mode_ffn = getattr(args, "mode_ffn", False)
    args.mode_2d = getattr(args, "mode_2d", False)
    args.mode_3d = getattr(args, "mode_3d", False)
    base_architecture(args)