import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from einops import rearrange
from data_processing.dataset import get_dataset_and_dataloader_all

class CNNBackbone_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNNBackbone_2, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv0 = nn.Conv2d(self.in_channel, 16, kernel_size=(3, 8), stride=(1, 4), padding=(1, 1), bias=False)
        self.batch_norm0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(3, 8), stride=(1, 4), padding=(1, 1), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(1, 4), padding=(1, 1), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, self.out_channel, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1), bias=False)
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

if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    # Test CNN backbone
    num_stacked_seqs = 4
    cnn = CNNBackbone_2(num_stacked_seqs).float()
    d_model = 128
    max_num_embedding = 10000
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=16, num_workers=0, num_stacked_seqs=num_stacked_seqs, mode='test')
    data_it = iter(dataloader)
    radar, label = next(data_it)
    embedding = cnn(radar)
    embedding = rearrange(embedding, 'b c a s -> b (a s) c')
    num_embedding = embedding.shape[1]
    pass
