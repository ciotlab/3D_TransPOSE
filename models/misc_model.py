import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from einops import rearrange
from data_processing.dataset import get_dataset_and_dataloader_all


def positional_encoding_sine(num_embedding, d_model, max_num_embedding, normalize, scale):
    seq_embed = torch.arange(1, num_embedding+1)
    if normalize:
        eps = 1e-6
        if scale is None:
            scale = 2 * math.pi * max_num_embedding
        seq_embed = seq_embed / (seq_embed[-1] + eps) * scale
    dim_t = torch.arange(d_model)
    dim_t = max_num_embedding ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / d_model)

    pe = seq_embed[:, None] / dim_t
    pe = torch.stack((pe[:, 0::2].sin(), pe[:, 1::2].cos()), dim=2).flatten(1)
    return pe


class CNNBackbone(nn.Module):
    def __init__(self, in_channel, out_channel=128):
        super(CNNBackbone, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv0 = nn.Conv2d(self.in_channel, 16, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, self.out_channel, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.out_channel)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        return x

class CNNBackbone_2(nn.Module):
    def __init__(self, in_channel, out_channel=128):
        super(CNNBackbone_2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv0 = nn.Conv2d(self.in_channel, 16, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, self.out_channel, kernel_size=(1, 4), stride=(1, 2), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(self.out_channel)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    # Test CNN backbone
    num_stacked_seqs = 4
    cnn = CNNBackbone(num_stacked_seqs).float()
    d_model = 128
    max_num_embedding = 10000
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=16, num_workers=0, num_stacked_seqs=num_stacked_seqs, mode='test')
    data_it = iter(dataloader)
    radar, label = next(data_it)
    embedding = cnn(radar)
    embedding = rearrange(embedding, 'b c a s -> b (a s) c')
    num_embedding = embedding.shape[1]
    pos = positional_encoding_sine(num_embedding, d_model, max_num_embedding, True, None)
    embedding = embedding + pos
    pass




