# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from typing import Optional
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
)

torch.set_printoptions(profile="full")

@register_model("transformer_meloform")
class TransformerMeloFormModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        source_sent_ids,
        target_sent_ids,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        
        batch_size = source_sent_ids.shape[0]
        new_encoder_padding_mask = torch.ones(encoder_out.encoder_padding_mask.shape, dtype=torch.bool, device=encoder_out.encoder_padding_mask.device)
        for i in range(batch_size):
            target_ids = torch.unique(target_sent_ids[i])
            for j, target_id in enumerate(target_ids):
                if target_id >= 0:
                    new_encoder_padding_mask[i] &= torch.ne(source_sent_ids[i], target_id)
     
        encoder_out = encoder_out._replace(encoder_padding_mask = new_encoder_padding_mask)

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

@register_model_architecture('transformer_meloform', 'transformer_meloform')
def transformer_meloform(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    base_architecture(args)
