from typing import Optional
from dataclasses import dataclass


def get_item(schemes, idx):
    if schemes is None:
        return None
    len_schemes = len(schemes)
    if len_schemes > idx:
        return schemes[idx]
    if len_schemes == 1:
        return schemes[0]
    raise IndexError('idx (%d) is out of the length (%d) of the schemes' % (idx, len_schemes), schemes)


@dataclass(frozen=True)
class HeadAttentionScheme:
    head_sum2sum: Optional[tuple]
    head_sum2sum_self: str
    head_sum2con: Optional[tuple]
    head_sum2con_self: str
    head_con2sum: Optional[tuple]
    head_con2sum_self: str
    head_con2con: Optional[tuple]
    head_con2con_self: str
    head_con2con_causal: bool
    head_pref2pref_mode: str
    head_sum2pref_mode: str
    head_pref2sum_mode: str
    head_con2pref_mode: str
    head_pref2con_mode: str


class LayerAttentionScheme(object):
    def __init__(
        self,
        layer_sum2sum, layer_sum2sum_self,
        layer_sum2con, layer_sum2con_self,
        layer_con2sum, layer_con2sum_self,
        layer_con2con, layer_con2con_self,
        layer_con2con_causal,
        layer_pref2pref_mode,
        layer_sum2pref_mode,
        layer_pref2sum_mode,
        layer_con2pref_mode,
        layer_pref2con_mode,
        num_layer_heads,
    ):
        self.num_layer_heads = num_layer_heads

        self.heads_schemes = []
        for idx in range(self.num_layer_heads):
            self.heads_schemes.append(
                HeadAttentionScheme(
                    get_item(layer_sum2sum, idx), layer_sum2sum_self,
                    get_item(layer_sum2con, idx), layer_sum2con_self,
                    get_item(layer_con2sum, idx), layer_con2sum_self,
                    get_item(layer_con2con, idx), layer_con2con_self,
                    layer_con2con_causal,
                    layer_pref2pref_mode,
                    layer_sum2pref_mode,
                    layer_pref2sum_mode,
                    layer_con2pref_mode,
                    layer_pref2con_mode
                )
            )
        self.heads_schemes = tuple(self.heads_schemes)
        self.same_for_all_heads = len(set(self.heads_schemes)) == 1

    def __hash__(self):
        return hash(self.heads_schemes)

    def __eq__(self, other):
        return self.heads_schemes == other.heads_schemes

    def __getitem__(self, idx):
        return self.heads_schemes[idx]

    def __len__(self):
        return self.num_layer_heads


class AttentionScheme(object):
    def __init__(
        self,
        sum2sum, sum2sum_self,
        sum2con, sum2con_self,
        con2sum, con2sum_self,
        con2con, con2con_self,
        con2con_causal,
        pref2pref_mode,
        sum2pref_mode,
        pref2sum_mode,
        con2pref_mode,
        pref2con_mode,
        num_layers, num_layers_heads,
    ):
        self.num_layers = num_layers

        self.layers_schemes = []
        for idx in range(self.num_layers):
            self.layers_schemes.append(
                LayerAttentionScheme(
                    get_item(sum2sum, idx), sum2sum_self,
                    get_item(sum2con, idx), sum2con_self,
                    get_item(con2sum, idx), con2sum_self,
                    get_item(con2con, idx), con2con_self,
                    con2con_causal,
                    pref2pref_mode,
                    sum2pref_mode,
                    pref2sum_mode,
                    con2pref_mode,
                    pref2con_mode,
                    get_item(num_layers_heads, idx)
                )
            )
        self.layers_schemes = tuple(self.layers_schemes)

    def __hash__(self):
        return hash(self.layers_schemes)

    def __eq__(self, other):
        return self.layers_schemes == other.layers_schemes

    def __getitem__(self, idx):
        return self.layers_schemes[idx]

    def __len__(self):
        return self.num_layers
