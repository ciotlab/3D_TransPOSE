import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, generalized_box_3d_iou
import logging
from data_processing.dataset import get_dataset_and_dataloader_all
from models.rddetr_2 import RDDETR_2
from models.matcher import HungarianMatcher
from util.box_ops import iou, box_cxcywh_to_xyxy, generalized_iou
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SetCriterion_2(nn.Module):
    def __init__(self, anchor, grid_size, device, threshold):
        super().__init__()
        self.anchor = anchor
        self.grid_size = 8
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.threshold = threshold
        self.device = device

    def forward(self, outputs, targets):
        batch_size, grid_size, _, _ = outputs.shape
        out_boxes = outputs[:, :, :, 0:4]
        out_confidence = outputs[:, :, :, 4]
        out_keypoints = outputs[:, :, :, 5:]

        x, y, w, h = transform_bbox(out_boxes, self.anchor)

        rddetr_targets = get_targets(out_boxes, targets, self.anchor, self.device, self.threshold)

        # target_boxes = rddetr_targets["tboxes"]
        tx = rddetr_targets["tx"]
        ty = rddetr_targets["ty"]
        tw = rddetr_targets["tw"]
        th = rddetr_targets["th"]
        tkey = rddetr_targets["tkey"]
        t_confidence = rddetr_targets["t_conf"]
        obj_mask = rddetr_targets["obj_mask"]
        noobj_mask = rddetr_targets["noobj_mask"]

        loss_giou = 0
        box1 = box_cxcywh_to_xyxy(x[obj_mask], y[obj_mask], w[obj_mask], h[obj_mask])
        box2 = box_cxcywh_to_xyxy(tx[obj_mask], ty[obj_mask], tw[obj_mask], th[obj_mask])

        loss_x = F.l1_loss(x[obj_mask], tx[obj_mask])
        loss_y = F.l1_loss(y[obj_mask], ty[obj_mask])
        loss_w = F.l1_loss(w[obj_mask], tw[obj_mask])
        loss_h = F.l1_loss(h[obj_mask], th[obj_mask])

        loss_conf_obj = self.bce_loss(out_confidence[obj_mask], t_confidence[obj_mask])
        loss_conf_noobj = self.bce_loss(out_confidence[noobj_mask], t_confidence[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_keypoint = F.l1_loss(out_keypoints[obj_mask], tkey[obj_mask], reduction='none')
        loss_giou = 1- generalized_iou(box1, box2)

        losses = {}
        losses['loss_boxes'] = loss_x + loss_y + loss_w + loss_h / tx[obj_mask].shape[0]
        losses['loss_keypoints'] = loss_keypoint.sum() / tkey[obj_mask].shape[0]
        losses['loss_conf'] = loss_conf / 100 * t_confidence[obj_mask].shape[0]
        losses['loss_iou'] = loss_giou.sum() / tx[obj_mask].shape[0]

        return losses


def transform_bbox(bbox, anchors):
    x = bbox[:, :, :, 0]
    y = bbox[:, :, :, 1]
    w = bbox[:, :, :, 2]
    h = bbox[:, :, :, 3]

    anchor_w = anchors[0].view((1, 1, 1))
    anchor_h = anchors[1].view((1, 1, 1))

    x = x - x.floor()
    y = y - y.floor()
    w = torch.log(w / anchor_w + 1e-16)
    h = torch.log(h / anchor_h + 1e-16)

    return x, y, w, h
    # return torch.stack((x, y, w, h))


def get_targets(pred_boxes, target, anchors, device, ignore_thresh):
    batch_size = pred_boxes.size(0)
    grid_size = pred_boxes.size(1)

    size_t = batch_size, grid_size, grid_size
    size_key = batch_size, grid_size, grid_size, 63

    obj_mask = torch.zeros(size_t, device=device, dtype=torch.bool)
    noobj_mask = torch.ones(size_t, device=device, dtype=torch.bool)
    tx = torch.zeros(size_t, device=device, dtype=torch.float32)
    ty = torch.zeros(size_t, device=device, dtype=torch.float32)
    tw = torch.zeros(size_t, device=device, dtype=torch.float32)
    th = torch.zeros(size_t, device=device, dtype=torch.float32)
    tkey = torch.zeros(size_key, device=device, dtype=torch.float32)

    for bs in range(batch_size):
        target_boxes = torch.tensor(target['boxes'][bs].reshape(-1, 6)).float().to(pred_boxes.device)
        target_keypoints = torch.tensor(target['keypoints'][bs].reshape(-1, 63)).float().to(pred_boxes.device)
        target_xy = target_boxes[:, :2] * grid_size
        target_wh = target_boxes[:, 3:5] * grid_size
        t_x, t_y = target_xy.t()
        t_w, t_h = target_wh.t()

        for o in range(len(t_x)):
            if t_x[o] > 8:
                t_x[o] = 7.9999
            elif t_y[0] > 8:
                t_y[0] = 7.9999
        t_w, t_h = target_wh.t()

        grid_i, grid_j = target_xy.long().t()

        save_file = Path(target['label_files'][bs])
        parts = list(save_file.parts)
        # print(parts[-1])

        for i in range(target_boxes.shape[0]):
            obj_mask[bs, grid_j[i], grid_i[i]] = 1
            noobj_mask[bs, grid_j[i], grid_i[i]] = 0

            tx[bs, grid_j[i], grid_i[i]] = t_x[i] - t_x[i].floor()
            ty[bs, grid_j[i], grid_i[i]] = t_y[i] - t_y[i].floor()

            tw[bs, grid_j[i], grid_i[i]] = torch.log(t_w[i] / anchors[0] + 1e-16)
            th[bs, grid_j[i], grid_i[i]] = torch.log(t_h[i] / anchors[1] + 1e-16)
            for j in range(target_keypoints.size(1)):
                tkey[bs, grid_j[i], grid_i[i], j] = target_keypoints[i, j]


    output = {
        "obj_mask": obj_mask,
        "noobj_mask": noobj_mask,
        # "tboxes" : torch.stack((tx, ty, tw, th)),
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "tkey": tkey,
        "t_conf": obj_mask.float(),
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
