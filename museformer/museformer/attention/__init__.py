from ..tools import arg_tools


def add_args(parser):
    parser.add_argument('--attn-query-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-key-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-value-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-out-proj-bias', type=arg_tools.str_bool_with_default_error)

    # valid: sum_then_reg, v2.1
    parser.add_argument('--attn-sum-key2-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-sum-value2-proj-bias', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-key2-value2-proj-weight', type=arg_tools.str_bool_with_default_error)

    parser.add_argument('--add-different-kqv-bias-for-sum-and-reg', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--add-different-out-bias-for-sum-and-reg', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-query-proj', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-key-proj', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-value-proj', type=arg_tools.str_bool_with_default_error)
    parser.add_argument('--attn-share-out-proj', type=arg_tools.str_bool_with_default_error)


def create_attention_v2_s1(
    *args, implementation='mask', block_size=64, **kwargs
):
    if implementation == 'blocksparse':
        from .self_attention_v2s1.blocksparse_rpe_self_attention_v2s1 import BlocksparseRpeSelfAttentionV2S1
        return BlocksparseRpeSelfAttentionV2S1(
            *args, block_size=block_size, **kwargs
        )
    return NotImplementedError(implementation)


def create_attention(
    *args, attention_mode='v2s1', **kwargs
):
    if attention_mode == 'v2s1':  # v2.1
        return create_attention_v2_s1(*args, **kwargs)
    raise NotImplementedError(attention_mode)
