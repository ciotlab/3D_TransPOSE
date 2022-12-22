import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from models.misc_model import positional_encoding_sine, CNNBackbone, MLP
from models.transformer import Transformer
import logging
from data_processing.dataset import get_dataset_and_dataloader_all
from models.backbone import resnet18, resnet34
from models.backbone_2 import CNNBackbone_2
from einops import rearrange, repeat


class RDDETR_2(nn.Module):
    def __init__(self, anchors):
        super(RDDETR_2, self).__init__()
        self.anchors = anchors
        self.num_keypoints = 63
        self.backbone = resnet18()
        self.backbone_2 = CNNBackbone_2(in_channel=8, out_channel=70)
        self.backbone_3 = resnet34()

    def forward(self, inputs):
        # transform signal shape to virtual array
        x = repeat(inputs, 'b f (t r) s -> (b f) t r s', t=8, r=8)
        inputs1 = x[:, 0:4, 4:8, :].clone().detach()
        inputs2 = x[:, 4:8, 0:4, :].clone().detach()
        x[:, 0:4, 4:8, :] = inputs2
        x[:, 4:8, 0:4, :] = inputs1
        x = rearrange(x, 'b t r s -> b r t s')

        # backbone
        # x = self.backbone(x)
        x = self.backbone_2(x)
        # x = self.backbone_3(x)
        batch_size = x.size(0)
        grid_size = x.size(2)

        prediction = x.view(batch_size, 7 + self.num_keypoints, grid_size, grid_size)
        prediction = prediction.permute(0, 3, 2, 1).contiguous()

        # prediction : ( bs, grid_size, grid_size, 70 )
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        z = torch.sigmoid(prediction[..., 2])

        w = prediction[..., 3]
        h = prediction[..., 4]
        d = prediction[..., 5]

        pred_boxes = torch.stack((x, y, z, w, h, d), -1)   # ( bs, grid_size, grid_size, 6[x,y,z,w,h,d] )
        obj_score = prediction[..., 6]  # ( bs, grid_size, grid_size, 1 )
        # pred_keypoint = torch.sigmoid(prediction[..., 7:])  # ( bs, grid_size, grid_size, 63 )
        pred_keypoint = prediction[..., 7:]
        output = {'pred_keypoints': pred_keypoint.view(batch_size, grid_size, grid_size, 63),
                  'pred_boxes': pred_boxes.view(batch_size, grid_size, grid_size, 6),
                  'pred_confidence': obj_score.view(batch_size, grid_size, grid_size, 1)}

        return output


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
    anchor = torch.tensor([3, 3]).to(device)
    rddetr_2 = RDDETR_2(anchor).to(device)
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=batch_size, num_workers=num_dataset_workers,
                                                         num_stacked_seqs=num_stacked_seqs, mode='test')
    logging.info(f'Parameters: {sum(p.numel() for p in rddetr_2.parameters() if p.requires_grad)}')
    iter_per_epoch = int(len(dataset) // batch_size)
    pbar = tqdm(enumerate(dataloader), total=iter_per_epoch, desc="Testing")
    for iter, (radar, label) in pbar:
        prediction = rddetr_2(radar.to(device))
