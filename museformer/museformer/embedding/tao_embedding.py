import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_embedding = init_size
        weights = self.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.register_buffer('weights', weights, persistent=False)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx=None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, x):
        if x.numel() > 0:
            x_max = int(x.max())
            if x_max >= self.num_embedding:
                self.num_embedding = x_max + 32
                weights = self.get_embedding(
                    self.num_embedding, self.embedding_dim, self.padding_idx
                )
                self.weights = weights.to(self.weights)
        r = F.embedding(x, self.weights, padding_idx=self.padding_idx)
        return r


def TaoEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
):
    if learned:
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings,
        )
    return m
