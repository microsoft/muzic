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
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy_with_alignment import \
    LabelSmoothedCrossEntropyCriterionWithAlignment

@register_criterion("meloform_label_smoothed_cross_entropy_with_alignment")
class MeloFormLabelSmoothedCrossEntropyCriterionWithAlignment(
    LabelSmoothedCrossEntropyCriterionWithAlignment
):

    def __init__(self, task, sentence_avg, label_smoothing, alignment_lambda):
        super().__init__(task, sentence_avg, label_smoothing, alignment_lambda)

    def compute_alignment_loss(self, sample, net_output):
        attn_prob = net_output[1]["attn"][0]
        bsz, tgt_sz, src_sz = attn_prob.shape
        attn = attn_prob.view(bsz * tgt_sz, src_sz)  # (batch * tgt_len, src_len)

        align = sample["alignments"]  # (:, 2)
        align_weights = sample["align_weights"].float()  # (:)

        if len(align) > 0:
            # Alignment loss computation. align (shape [:, 2]) contains the src-tgt index pairs corresponding to
            # the alignments. align_weights (shape [:]) contains the 1 / frequency of a tgt index for normalizing.
            loss = -(
                (
                    attn[
                        align[:, 1][:, None],  # (:, 1, 1)
                        align[:, 0][:, None]   # (:, 1, 1)
                    ] * (1 - 10e-5) + 10e-5
                ).log()  # (: 1)
                * align_weights[:, None]  # (:, 1)
            ).sum()
        else:
            return None

        try:
            assert not loss.isnan().any()
            assert not loss.isinf().any()
        except AssertionError:
            print('align')
            print(align)
            print('align_weights')
            print(align_weights)
            print('loss at alignments')
            print(attn[align[:, 1][:, None], align[:, 0][:, None]])
            print('loss')
            print(loss)
            print('attn_prob')
            print(attn_prob)
            raise

        return loss