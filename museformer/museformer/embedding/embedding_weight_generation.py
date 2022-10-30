import torch
import torch.nn as nn


def generate_sinusoid_embeddings(max_len, dim):
    inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
    position = torch.arange(max_len)  # (max_len, 1)
    sinusoid_inp = torch.ger(position, inv_freq)
    pe = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pe


def generate_sinusoid_position_embedding_with_padding(num_positions, embed_dim):
    embedding_weight = generate_sinusoid_embeddings(num_positions, embed_dim)
    embedding_weight = torch.cat(
        (torch.zeros(1, embed_dim), embedding_weight), dim=0
    )
    return embedding_weight


def generate_randomly_initialized_position_embedding_with_padding(num_positions, embed_dim):
    embedding_weight = torch.empty(num_positions + 1, embed_dim)
    nn.init.normal_(embedding_weight)
    nn.init.zeros_(embedding_weight[0])
    return embedding_weight
