import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from models.misc_model import positional_encoding_sine, CNNBackbone, MLP
from models.transformer import Transformer
import logging
from data_processing.dataset import get_dataset_and_dataloader_all
from models.backbone import resnet18


class RDDETR_2(nn.Module):
    def __init__(self, anchors, device):
        super(RDDETR_2, self).__init__()
        self.anchors = anchors
        self.num_anchors = 1
        self.num_keypoints = 63
        self.grid_size = 0
        self.backbone = resnet18()
        self.device = device

    def forward(self, inputs):
        x = self.backbone(inputs)
        # x = repeat(x, 'b (x y s) -> b x y s', x=7, y=7, s=68)
        batch_size = x.size(0)
        grid_size = x.size(2)

        prediction = x.view(batch_size, self.num_anchors, 5 + self.num_keypoints, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        obj_score = torch.sigmoid(prediction[..., 4])
        pred_keypoint = torch.sigmoid(prediction[..., 5:])

        self.compute_grid_offsets(grid_size)
        pred_boxes = self.transform_outputs(prediction)

        output = torch.cat((pred_boxes.view(batch_size, grid_size, grid_size, 4),
                            obj_score.view(batch_size, grid_size, grid_size, 1),
                            pred_keypoint.view(batch_size, grid_size, grid_size, self.num_keypoints)), -1)

        return output

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        self.stride = self.anchors[0] / self.grid_size
        self.grid_x = torch.arange(grid_size).repeat(1, 1, grid_size, 1).type(torch.float32).to(self.device)
        self.grid_y = torch.arange(grid_size).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32).to(self.device)

        scaled_anchors = self.anchors[0] / self.stride, self.anchors[0] / self.stride
        self.scaled_anchors = torch.tensor(scaled_anchors, device=self.device)
        self.anchor_w = self.scaled_anchors[0].view(1, 1, 1, 1)
        self.anchor_h = self.scaled_anchors[1].view(1, 1, 1, 1)

    def transform_outputs(self, prediction):
        x, y = torch.sigmoid(prediction[..., 0]), torch.sigmoid(prediction[..., 1])
        w, h = prediction[..., 2], prediction[..., 3]

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(self.device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        return pred_boxes * self.stride


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
    rddetr_2 = RDDETR_2(anchor, device).to(device)
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=batch_size, num_workers=num_dataset_workers,
                                                         num_stacked_seqs=num_stacked_seqs, mode='test')
    logging.info(f'Parameters: {sum(p.numel() for p in rddetr_2.parameters() if p.requires_grad)}')
    iter_per_epoch = int(len(dataset) // batch_size)
    pbar = tqdm(enumerate(dataloader), total=iter_per_epoch, desc="Testing")
    for iter, (radar, label) in pbar:
        prediction = rddetr_2(radar.to(device))
