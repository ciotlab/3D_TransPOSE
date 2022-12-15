import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from einops import rearrange
from models.misc_model import positional_encoding_sine, CNNBackbone, MLP
from models.transformer import Transformer
import logging
from data_processing.dataset import get_dataset_and_dataloader_all


class RDDETR(nn.Module):
    def __init__(self, num_stacked_seqs, d_model, num_queries, n_head, num_layers, dim_feedforward,
                 dropout, activation):
        super(RDDETR, self).__init__()
        self.num_stacked_seqs = num_stacked_seqs
        self.d_model = d_model
        self.num_queries = num_queries
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.backbone = CNNBackbone(in_channel=num_stacked_seqs, out_channel=d_model)
        self.transformer = Transformer(d_model=d_model, n_head=n_head, num_layers=num_layers,
                                       dim_feedforward=dim_feedforward, dropout=dropout,
                                       activation=activation, return_intermediate=True)
        self.box_head = MLP(input_dim=self.d_model, hidden_dim=self.d_model, output_dim=6, num_layers=3)
        self.keypoint_dist_head = nn.Linear(self.d_model, 21 * 3)
        self.objectness_head = nn.Linear(self.d_model, 1)
        self.query = nn.Embedding(self.num_queries, self.d_model)
        self.query_pos_embedding = nn.Embedding(self.num_queries, self.d_model)
        source_pos = positional_encoding_sine(num_embedding=10000, d_model=d_model, max_num_embedding=10000,
                                              normalize=False, scale=None)
        self.register_buffer('source_pos', source_pos)
        query_zero = torch.zeros(self.num_queries, self.d_model)
        self.register_buffer('query_zero', query_zero)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = rearrange(x, 'b c a s -> b (a s) c')
        source_pos = self.source_pos[:x.shape[1], :]
        x, attn, intermediate_output, intermediate_attn = self.transformer(self.query_zero, x, self.query_pos_embedding.weight, source_pos)
        output_3d_box = self.box_head(x).reshape(-1, self.num_queries, 2, 3).sigmoid()
        output_keypoint_dist = self.keypoint_dist_head(x).reshape(-1, self.num_queries, 21, 3).sigmoid()
        output_confidence_logit = self.objectness_head(x)
        output_confidence = output_confidence_logit.sigmoid()
        prediction = {'pred_keypoints': output_keypoint_dist, 'pred_boxes': output_3d_box,
                      'attention_map': intermediate_attn + [attn], 'pred_confidence_logit': output_confidence_logit,
                      'pred_confidence': output_confidence}
        return prediction


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    num_stacked_seqs = 1
    d_model = 128
    num_queries = 100
    n_head = 8
    num_layers = 3
    dim_feedforward = 2048
    dropout = 0.1
    activation = 'gelu'
    device = 'cuda:0'
    batch_size = 2
    num_dataset_workers = 4
    rddetr = RDDETR(num_stacked_seqs, d_model, num_queries, n_head, num_layers, dim_feedforward,
                    dropout, activation).to(device)
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=batch_size, num_workers=num_dataset_workers,
                                                     num_stacked_seqs=num_stacked_seqs, mode='test')
    logging.info(f'Parameters: {sum(p.numel() for p in rddetr.parameters() if p.requires_grad)}')
    iter_per_epoch = int(len(dataset) // batch_size)
    pbar = tqdm(enumerate(dataloader), total=iter_per_epoch, desc="Testing")
    for iter, (radar, label) in pbar:
        prediction = rddetr(radar.to(device))






