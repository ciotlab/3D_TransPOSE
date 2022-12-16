import torch
import numpy as np


def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    box1_x1 = x1 - w1 / 2
    box1_y1 = y1 - h1 / 2
    box1_x2 = x1 + w1 / 2
    box1_y2 = y1 + h1 / 2

    box2_x1 = x2 - w2 / 2
    box2_y1 = y2 - h2 / 2
    box2_x2 = x2 + w2 / 2
    box2_y2 = y2 + h2 / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-7)


def iou_eval(x1, y1, w1, h1, x2, y2, w2, h2):
    box1_x1 = x1 - w1 / 2
    box1_y1 = y1 - h1 / 2
    box1_x2 = x1 + w1 / 2
    box1_y2 = y1 + h1 / 2

    box2_x1 = x2 - w2 / 2
    box2_y1 = y2 - h2 / 2
    box2_x2 = x2 + w2 / 2
    box2_y2 = y2 + h2 / 2

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-7)


def generalized_iou(pr_bboxes, gt_bboxes):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    return iou - (enclosure - union) / enclosure


def box_cxcywh_to_xyxy(cx, cy, w, h):
    new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new, dim=-1)


def box_3d_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def box_3d_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2,
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)


def box_3d_iou(boxes1, boxes2):
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    maxmin = torch.max(boxes1[:, None, :3], boxes2[:, :3])
    minmax = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])

    wlh = (minmax - maxmin).clamp(min=0)
    inter = wlh[:, :, 0] * wlh[:, :, 1] * wlh[:, :, 2]

    union = volume1[:, None] + volume2 - inter

    iou = inter / union
    return iou, union


def generalized_box_3d_iou(boxes1, boxes2):
    iou, union = box_3d_iou(boxes1, boxes2)

    minmin = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    maxmax = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wlh = (maxmax - minmin).clamp(min=0)
    volume = wlh[:, :, 0] * wlh[:, :, 1] * wlh[:, :, 2]

    return iou - (volume - union) / volume


def get_iou_wh(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area
