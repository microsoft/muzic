from fairseq.models.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig, \
    DEFAULT_MAX_TARGET_POSITIONS, transformer_lm_gpt, base_lm_architecture
from fairseq import options
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.models import register_model, register_model_architecture

from .transformer import LinearTransformerDecoder

@register_model("linear_transformer_lm", dataclass=TransformerLanguageModelConfig)
class LinearTransformerLanguageModel(TransformerLanguageModel):
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--gradient-checkpointing', type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument('--gradient-checkpointing-every-n-layer', type=int, default=1)
        parser.add_argument('--gradient-checkpointing-layers',
                            type=lambda x: tuple([int(item) for item in x.split(',')]), default=None)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = LinearTransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)


@register_model_architecture("linear_transformer_lm", "linear_transformer_lm_gpt")
def linear_transformer_lm_gpt_architecture(args):
    transformer_lm_gpt(args)


@register_model_architecture("linear_transformer_lm", "linear_transformer_lm_std")
def std_linear_transformer_lm_architecture(args):
    args.command_embed_dim = 16
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)

@register_model_architecture("linear_transformer_lm", "linear_transformer_lm_debug")
def std_linear_transformer_lm_architecture(args):
    args.command_embed_dim = 16
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 32)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 32)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
