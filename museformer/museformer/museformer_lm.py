from dataclasses import dataclass
from typing import Optional
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqLanguageModel, register_model, register_model_architecture

from .tools import arg_tools
from .museformer_decoder import MuseformerDecoder

DEFAULT_MAX_TARGET_POSITIONS = 100000


@dataclass
class MuseformerLanguageModelConfig(FairseqDataclass):
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    eob_token: str = II("task.eob_token")
    eoc_token: str = II("task.eoc_token")
    chunking_scheme: str = II("task.chunking_scheme")
    fixed_chunking_length = int = II("task.fixed_chunking_length")


@register_model('museformer_lm', dataclass=MuseformerLanguageModelConfig)
class MuseformerLanguageModel(FairseqLanguageModel):
    _submodules = (MuseformerDecoder,)

    @classmethod
    def add_args(cls, parser):
        super(MuseformerLanguageModel, cls).add_args(parser)
        arg_tools.add_submodule_args(cls, parser)

    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        decoder = MuseformerDecoder(args, task.target_dictionary)
        print(args)
        return cls(decoder)

    def get_targets(self, sample, net_output):
        return self.decoder.get_targets(sample, net_output)


def base_lm_scheme(args):
    args.pref2pref_mode = getattr(args, "pref2pref_mode", 'lt')
    args.sum2pref_mode = getattr(args, "sum2pref_mode", 'none')
    args.pref2sum_mode = getattr(args, "pref2sum_mode", "none")
    args.con2pref_mode = getattr(args, "con2pref_mode", 'full')
    args.pref2con_mode = getattr(args, "pref2con_mode", 'none')


def b1248_lm_scheme(args):
    args.con2con = getattr(args, "con2con", ((((-2, 0), -4, -8, -12, -16, -24, -32),),))
    args.con2con_self = getattr(args, "con2con_self", "full")
    args.con2con_causal = getattr(args, "con2con_causal", True)
    args.con2sum = getattr(args, "con2sum",
                           ((((None, -32), (-31, -24), (-23, -16), (-15, -12), (-11, -8), (-7, -4), -3,),),))
    args.con2sum_self = getattr(args, "con2sum_self", "none")
    args.sum2con = getattr(args, "sum2con", ((None,),))
    args.sum2con_self = getattr(args, "sum2con_self", "full")
    args.sum2sum = getattr(args, "sum2sum", ((None,),))
    args.sum2sum_self = getattr(args, "sum2sum_self", "full")
    base_lm_scheme(args)


def b1234_lm_scheme(args):
    args.con2con = getattr(args, "con2con", ((((-8, 0),),),))
    args.con2con_self = getattr(args, "con2con_self", "full")
    args.con2con_causal = getattr(args, "con2con_causal", True)
    args.con2sum = getattr(args, "con2sum", ((((None, -8),),),))
    args.con2sum_self = getattr(args, "con2sum_self", "none")
    args.sum2con = getattr(args, "sum2con", ((None,),))
    args.sum2con_self = getattr(args, "sum2con_self", "full")
    args.sum2sum = getattr(args, "sum2sum", ((None,),))
    args.sum2sum_self = getattr(args, "sum2sum_self", "full")
    base_lm_scheme(args)


def share_sum_reg_params(args):
    args.share_layernorm_embedding = getattr(args, 'share_layernorm_embedding', True)
    args.attn_share_query_proj = getattr(args, 'attn_share_query_proj', True)
    args.attn_share_key_proj = getattr(args, 'attn_share_key_proj', True)
    args.attn_share_value_proj = getattr(args, 'attn_share_value_proj', True)
    args.attn_share_out_proj = getattr(args, 'attn_share_out_proj', True)
    args.share_self_attention_layer_norm = getattr(args, 'share_self_attention_layer_norm', True)
    args.share_ffn = getattr(args, 'share_ffn', True)
    args.share_final_layer_norm = getattr(args, 'share_final_layer_norm', True)


def base_lm_architecture(args):
    args.attention_embed_dim = getattr(args, "attention_embed_dim", 512)
    args.num_layers = getattr(args, "num_layers", 4)
    args.num_attention_heads = getattr(args, "num_attention_heads", (8,))
    args.normalize_before = getattr(args, "normalize_before", True)
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 2048)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.no_final_norm = getattr(args, "no_final_norm", False)

    args.take_bos_as_bar = getattr(args, 'take_bos_as_bar', True)
    args.num_summary_tokens_each_chunk = getattr(args, "num_summary_tokens_each_chunk", 1)

    args.share_input_output_embed = getattr(args, "share_input_output_embed", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)

    args.token_embed_dim = getattr(args, "token_embed_dim", args.attention_embed_dim)

    if getattr(args, "max_target_positions", None) is None:
        args.max_target_positions = getattr(
            args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
        )
    if args.num_summary_tokens_each_chunk <= 0:
        args.num_summary_tokens_each_chunk = 0
        args.con2sum = ((None,),)
        args.sum2con = ((None,),)
        args.sum2sum = ((None,),)

    args.attention_impl = getattr(args, "attention_impl", "blocksparse")
    if args.attention_impl == 'blocksparse':
        args.block_size = getattr(args, "block_size", 64)


@register_model_architecture("museformer_lm", "museformer_lm_v2s1")
def museformer_lm_v2s1(args):
    args.attention_mode = getattr(args, 'attention_mode', 'v2s1')

    b1248_lm_scheme(args)
    share_sum_reg_params(args)

    args.tokens_per_sample = getattr(args, 'tokens_per_sample', 100000)

    args.use_token_in_chunk_abs_pos = getattr(args, 'use_token_in_chunk_abs_pos', False)
    args.use_token_abs_pos = getattr(args, 'use_token_abs_pos', False)
    args.use_bar_abs_pos = getattr(args, 'use_bar_abs_pos', True)
    args.max_bar_abs_pos = getattr(args, 'max_bar_abs_pos', 512)
    args.bar_abs_pos_embed_dim = getattr(args, 'bar_abs_pos_embed_dim', 256)
    args.use_beat_abs_pos = getattr(args, 'use_beat_abs_pos', True)
    args.max_beat_abs_pos = getattr(args, 'max_beat_abs_pos', 64)
    args.beat_abs_pos_embed_dim = getattr(args, 'beat_abs_pos_embed_dim', 128)

    args.concat_reg_embeddings = getattr(args, 'concat_reg_embeddings', True)
    args.proj_reg_embeddings = getattr(args, 'proj_reg_embeddings', True)
    args.concat_sum_embeddings = getattr(args, 'concat_sum_embeddings', True)
    args.proj_sum_embeddings = getattr(args, 'proj_sum_embeddings', True)

    args.attn_query_proj_bias = getattr(args, 'attn_query_proj_bias', True)
    args.attn_key_proj_bias = getattr(args, 'attn_key_proj_bias', True)
    args.sum_key2_proj_bias = getattr(args, 'attn_sum_key2_proj_bias', True)
    args.attn_value_proj_bias = getattr(args, 'attn_value_proj_bias', True)
    args.sum_value2_proj_bias = getattr(args, 'attn_sum_value2_proj_bias', True)
    args.attn_out_proj_bias = getattr(args, 'attn_out_proj_bias', True)
    args.add_different_kqv_bias_for_sum_and_reg = getattr(args, 'add_different_kqv_bias_for_sum_and_reg', True)

    base_lm_architecture(args)
