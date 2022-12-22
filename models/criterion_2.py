import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
import logging
from data_processing.dataset import get_dataset_and_dataloader_all
from models.rddetr_2 import RDDETR_2
from util.box_ops import iou, box_iou, box_cxcywh_to_xyxy, matching_box_iou
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SetCriterion_2(nn.Module):
    def __init__(self, anchor, empty_weight, device):
        super().__init__()
        self.register_buffer('empty_weight', torch.tensor(empty_weight))
        self.anchor = anchor
        self.device = device

    def forward(self, outputs, targets):
        out_boxes = outputs['pred_boxes']
        out_confidence = outputs['pred_confidence'].squeeze(-1)
        out_keypoints = outputs['pred_keypoints']
        x, y, z, w, h, d = [out_boxes[..., t] for t in range(6)]

        rddetr_targets = get_targets(out_boxes, targets, self.anchor, self.device)

        tx = rddetr_targets["tx"]
        ty = rddetr_targets["ty"]
        tw = rddetr_targets["tw"]
        th = rddetr_targets["th"]
        tz = rddetr_targets["tz"]
        td = rddetr_targets["td"]
        tkey = rddetr_targets["tkey"]
        t_confidence = rddetr_targets["t_conf"]
        obj_mask = rddetr_targets["obj_mask"]

        bs = out_boxes.size(0)
        grid = out_boxes.size(1)

        px = x + torch.arange(grid).repeat(grid, 1).view(1, grid, grid).to(self.device)
        py = y + torch.arange(grid).repeat(grid, 1).t().view(1, grid, grid).to(self.device)
        pw = torch.exp(w) * self.anchor[0]
        ph = torch.exp(h) * self.anchor[1]
        
        tmp_tx = rddetr_targets["tmp_tx"]
        tmp_ty = rddetr_targets["tmp_ty"]
        tmp_tw = rddetr_targets["tmp_tw"]
        tmp_th = rddetr_targets["tmp_th"]

        loss_x = F.l1_loss(x[obj_mask], tx[obj_mask])
        loss_y = F.l1_loss(y[obj_mask], ty[obj_mask])
        loss_w = F.l1_loss(w[obj_mask], tw[obj_mask])
        loss_h = F.l1_loss(h[obj_mask], th[obj_mask])
        loss_z = F.l1_loss(z[obj_mask], tz[obj_mask])
        loss_d = F.l1_loss(d[obj_mask], td[obj_mask])

        loss_conf_obj = F.binary_cross_entropy_with_logits(out_confidence, t_confidence, pos_weight=self.empty_weight,
                                                           reduction='none')
        loss_keypoint = F.l1_loss(out_keypoints[obj_mask], tkey[obj_mask], reduction='none')

        pbox = torch.stack((px[obj_mask], py[obj_mask], pw[obj_mask], ph[obj_mask]), dim=1)
        tbox = torch.stack((tmp_tx[obj_mask], tmp_ty[obj_mask], tmp_tw[obj_mask], tmp_th[obj_mask]), dim=1)

        iou = box_iou(box_cxcywh_to_xyxy(pbox), box_cxcywh_to_xyxy(tbox))[0]
        loss_iou = 1 - iou

        losses = {
            'loss_boxes': (loss_x + loss_y + loss_w + loss_h + loss_z + loss_d),
            'loss_keypoints': loss_keypoint.sum() / tkey[obj_mask].shape[0],
            'loss_conf': loss_conf_obj.mean(),
            'loss_iou': loss_iou.sum() / tbox.shape[0]
        }

        return losses


def get_targets(pred_boxes, target, anchor, device):
    batch_size = pred_boxes.size(0)
    grid_size = pred_boxes.size(1)

    size_t = batch_size, grid_size, grid_size
    size_key = batch_size, grid_size, grid_size, 63

    obj_mask = torch.zeros(size_t, device=device, dtype=torch.bool)
    tx = torch.zeros(size_t, device=device, dtype=torch.float32)
    ty = torch.zeros(size_t, device=device, dtype=torch.float32)
    tz = torch.zeros(size_t, device=device, dtype=torch.float32)
    tw = torch.zeros(size_t, device=device, dtype=torch.float32)
    th = torch.zeros(size_t, device=device, dtype=torch.float32)
    td = torch.zeros(size_t, device=device, dtype=torch.float32)

    tmp_tx = torch.zeros(size_t, device=device, dtype=torch.float32)
    tmp_ty = torch.zeros(size_t, device=device, dtype=torch.float32)
    tmp_tz = torch.zeros(size_t, device=device, dtype=torch.float32)
    tmp_tw = torch.zeros(size_t, device=device, dtype=torch.float32)
    tmp_th = torch.zeros(size_t, device=device, dtype=torch.float32)
    tmp_td = torch.zeros(size_t, device=device, dtype=torch.float32)

    tkey = torch.zeros(size_key, device=device, dtype=torch.float32)

    for bs in range(batch_size):
        target_boxes = torch.tensor(target['boxes'][bs].reshape(-1, 6)).float().to(pred_boxes.device)
        target_keypoints = torch.tensor(target['keypoints'][bs].reshape(-1, 63)).float().to(pred_boxes.device)
        target_xy = target_boxes[:, :2] * grid_size
        target_wh = target_boxes[:, 3:5] * grid_size
        t_x, t_y = target_xy.t()
        t_w, t_h = target_wh.t()
        t_z = target_boxes[:, 2]
        t_d = target_boxes[:, 5]

        grid_i, grid_j = target_xy.long().t()

        obj_mask[bs, grid_j, grid_i] = 1
        tx[bs, grid_j, grid_i] = t_x - t_x.floor()
        ty[bs, grid_j, grid_i] = t_y - t_y.floor()
        tz[bs, grid_j, grid_i] = t_z

        tw[bs, grid_j, grid_i] = torch.log(t_w / anchor[0] + 1e-16)
        th[bs, grid_j, grid_i] = torch.log(t_h / anchor[1] + 1e-16)
        td[bs, grid_j, grid_i] = torch.log(t_d + 1e-16)

        tmp_tx[bs, grid_j, grid_i] = t_x
        tmp_ty[bs, grid_j, grid_i] = t_y
        tmp_tw[bs, grid_j, grid_i] = t_w
        tmp_th[bs, grid_j, grid_i] = t_h

        for j in range(target_keypoints.size(1)):
            tkey[bs, grid_j, grid_i, j] = target_keypoints[:, j]

    output = {
        "obj_mask": obj_mask,
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "tkey": tkey,
        "t_conf": obj_mask.float(),
        "tmp_tx": tmp_tx,
        "tmp_ty": tmp_ty,
        "tmp_tw": tmp_tw,
        "tmp_th": tmp_th,
        "tz": tz,
        "td": td
    }

    return output


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    num_stacked_seqs = 1
    d_model = 128
    num_queries = 150
    n_head = 8
    num_layers = 3
    dim_feedforward = 2048
    dropout = 0.1
    activation = 'gelu'
    device = 'cuda:0'
    batch_size = 2
    num_dataset_workers = 2

    anchor = torch.tensor([3, 3]).to(device)
    grid_size = 8
    threshold = 0.5

    rddetr_2 = RDDETR_2(anchor, device).to(device)
    criterion = SetCriterion_2(anchor, grid_size, device, threshold).to(device)
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=batch_size, num_workers=num_dataset_workers,
                                                         num_stacked_seqs=num_stacked_seqs, mode='test')
    data_it = iter(dataloader)
    radar, label = next(data_it)
    prediction = rddetr_2(radar.to(device))
    loss = criterion(prediction, label)

    pass
