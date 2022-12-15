import copy
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import logging
from einops import rearrange
from data_processing.dataset import get_dataset_and_dataloader
from models.misc_model import CNNBackbone, positional_encoding_sine


class Transformer(nn.Module):
    def __init__(self, d_model=512, n_head=8, num_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", return_intermediate=False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        layer = TransformerLayer(d_model, n_head, dim_feedforward, dropout, activation)
        self.layers = _get_clones(layer, num_layers)
        self.return_intermediate = return_intermediate
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, source, query_pos, source_pos):
        num_batch = source.shape[0]
        query = query.expand(num_batch, -1, -1)
        query_pos = query_pos.expand(num_batch, -1, -1)
        source_pos = source_pos.expand(num_batch, -1, -1)
        intermediate_output = []
        intermediate_attn = []
        x = query
        attn = torch.empty(0)
        for i, layer in enumerate(self.layers):
            x, attn = layer(query=x, source=source, query_pos=query_pos, source_pos=source_pos)
            if self.return_intermediate and i < len(self.layers)-1:
                intermediate_output.append(x)
                intermediate_attn.append(attn)
        return x, attn, intermediate_output, intermediate_attn


class TransformerLayer(nn.Module):

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # Multihead attention modules
        self.self_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, source, query_pos, source_pos):
        x = query
        q = k = self.with_pos_embed(x, query_pos)
        x2, self_attn_map = self.self_attn(query=q, key=k, value=x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2, cross_attn_map = self.cross_attn(query=self.with_pos_embed(x, query_pos),
                                             key=self.with_pos_embed(source, source_pos),
                                             value=source)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x, cross_attn_map


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    num_stacked_seqs = 4
    d_model = 128
    num_queries = 200
    max_num_embedding = 10000
    cnn = CNNBackbone(in_channel=num_stacked_seqs).float()
    transformer = Transformer(d_model=d_model, n_head=8, num_layers=3, dim_feedforward=2048, dropout=0.1,
                              activation="gelu", return_intermediate=True).float()
    dataloader, dataset = get_dataset_and_dataloader(batch_size=16, num_workers=0, num_stacked_seqs=num_stacked_seqs, mode='test')
    data_it = iter(dataloader)
    radar, label = next(data_it)
    source = cnn(radar)
    source = rearrange(source, 'b c a s -> b (a s) c')
    num_source_embedding = source.shape[1]
    source_pos = positional_encoding_sine(max_num_embedding, d_model, max_num_embedding, False, None)
    source_pos = source_pos[:source.shape[1], :]
    query = torch.zeros(num_queries, d_model)
    query_pos_embedding = nn.Embedding(num_queries, d_model)
    output, attn, intermediate_output, intermediate_attn = transformer(query, source, query_pos_embedding.weight, source_pos)
    pass
