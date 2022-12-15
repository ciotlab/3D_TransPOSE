from typing import Iterable, Dict, List
import numpy as np
from pathlib import Path
import os
import mpl_toolkits.mplot3d.axes3d as p3
import torch
import wandb
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from einops import rearrange

import util.misc as utils
from models.criterion_2 import get_targets, transform_bbox
from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, box_3d_iou
from models.matcher import HungarianMatcher
from data_processing.dataset import area_min_xyz, area_size_xyz
from data_processing.dataset_visualization import tx_antenna_position, rx_antenna_position


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, use_wandb=False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        total_loss = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if use_wandb:
            wandb.log({"loss": total_loss, "loss_keypoints": loss_dict['loss_keypoints'],
                       "loss_boxes": loss_dict['loss_boxes'], "loss_iou": loss_dict['loss_iou'],
                       "loss_conf": loss_dict['loss_conf']})
        metric_logger.update(loss=total_loss, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, anchor, device, threshold, keypoint_thresh_list, nms_iou_thresh=0.5,
             matching_iou_thresh=0.5, conf_thresh=0.9, save_skeleton=False, save_attention_weight=False):
    model.to(device).eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    conf_matched_list = []
    total_num_target = 0
    skeleton_list = []
    keypoint_error_list = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        outputs = model(samples)
        # outputs = nms(outputs, iou_thresh=nms_iou_thresh)
        conf_matched, num_target, skeleton, keypoint_error \
            = get_batch_statistics(outputs, targets, anchor, device, threshold, matching_iou_thresh, conf_thresh, save_skeleton,
                                   save_attention_weight)
        conf_matched_list.append(conf_matched)
        total_num_target += num_target
        skeleton_list.extend(skeleton)
        keypoint_error_list.append(keypoint_error)
    conf_matched = torch.concat(conf_matched_list, dim=0)
    conf_matched = conf_matched.cpu().numpy()
    ap, pr_curve = compute_ap(conf_matched, total_num_target)
    keypoint_error = np.concatenate(keypoint_error_list, axis=0)
    keypoint_pck, per_keypoint_pck, keypoint_cdf, per_keypoint_cdf \
        = compute_pck(keypoint_error, keypoint_thresh_list)
    stats = {'AP': ap, 'pr_curve': pr_curve, 'keypoint_pck': keypoint_pck, 'per_keypoint_pck': per_keypoint_pck,
             'keypoint_cdf': keypoint_cdf, 'per_keypoint_cdf': per_keypoint_cdf, 'skeleton': skeleton_list}
    return stats


def nms(outputs, iou_thresh):
    batch_size, num_queries = outputs["pred_boxes"].shape[:2]
    for b in range(batch_size):
        out_boxes = rearrange(outputs['pred_boxes'][b], "q p c -> q (p c)")
        out_confidence = outputs['pred_confidence'][b, :, 0]
        sort_idx = torch.argsort(out_confidence, dim=0, descending=True).long()
        out_boxes = box_3d_cxcyczwhd_to_xyzxyz(out_boxes[sort_idx, :])
        out_confidence = out_confidence[sort_idx]
        iou, union = box_3d_iou(out_boxes, out_boxes)
        overlap = iou > iou_thresh
        for idx, conf in enumerate(out_confidence):
            if conf > 0:
                zero_idx = torch.full_like(out_confidence, False, dtype=torch.bool)
                zero_idx[idx+1:] = overlap[idx, idx+1:]
                out_confidence[zero_idx] = 0
        outputs['pred_confidence'][b, sort_idx, 0] = out_confidence
    return outputs


def get_batch_statistics(outputs, targets, anchor, device, threshold, iou_thresh, conf_thresh, save_skeleton=False,
                         save_attention_weight=False, ):

    pred_boxes = outputs[:, :, :, :4]
    pred_conf = outputs[:, :, :, 4]
    pred_keypoints = outputs[:, :, :, 5:]
    tgt_boxes = targets['boxes']
    tgt_keypoints = targets['keypoints']
    tgt_label_files = targets['label_files']
    conf_matched = []
    total_num_target = 0

    px, py, pw, ph = transform_bbox(pred_boxes, anchor)

    tgts = get_targets(pred_boxes, targets, anchor, device, threshold)
    tx = tgts["tx"]
    ty = tgts["ty"]
    tw = tgts["tw"]
    th = tgts["th"]
    tkey = tgts["tkey"]
    t_confidence = tgts["t_conf"]
    obj_mask = tgts["obj_mask"]
    noobj_mask = tgts["noobj_mask"]
    tmp_tx = tx[obj_mask]
    tmp_pbox = pred_boxes[obj_mask]

    # extract data for AP computation
    # for i, index in enumerate(indices):
    #     conf = pred_confidence[i]
    #     matched = torch.zeros_like(conf)
    #     matched[index[0]] = 1
    #     conf_matched.append(torch.concat((conf, matched), 1))
    #     num_target = tgt_boxes[i].shape[0]
    #     total_num_target += num_target
    # conf_matched = torch.concat(conf_matched, 0)

    # extract skeleton and compute keypoint errors
    skeleton = []
    keypoint_error = []

    p_conf = pred_conf[obj_mask].cpu().numpy()
    px = px[obj_mask].cpu().numpy()
    py = py[obj_mask].cpu().numpy()
    pw = pw[obj_mask].cpu().numpy()
    ph = ph[obj_mask].cpu().numpy()
    pkpt = pred_keypoints[obj_mask].cpu().numpy().reshape(-1, 21, 3)

    tx = tx[obj_mask].cpu().numpy()
    ty = ty[obj_mask].cpu().numpy()
    tw = tw[obj_mask].cpu().numpy()
    th = th[obj_mask].cpu().numpy()
    tkpt = tkey[obj_mask].cpu().numpy().reshape(-1, 21, 3)

    pbox = np.stack((px, py, pw, ph)).T
    tbox = np.stack((tx, ty, tw, th)).T

    for i in range(tx.shape[0]):
        pbox_center = pbox[i, :2] * area_size_xyz[:2] + area_min_xyz[:2]
        pbox_size = pbox[i, 2:] * area_size_xyz[:2]
        pbox_min = pbox_center - pbox_size / 2
        pkpt_orig = np.zeros(pkpt.shape)
        pkpt_orig[i][:, :2] = pkpt[i][:, :2] * pbox_size + pbox_min
        pkpt_orig[i][:, 2] = pkpt[i][:, 2] * area_size_xyz[2]

        tbox_center = tbox[i, :2] * area_size_xyz[:2] + area_min_xyz[:2]
        tbox_size = tbox[i, 2:] * area_size_xyz[:2]
        tbox_min = tbox_center - tbox_size / 2
        tkpt_orig = np.zeros(tkpt.shape)
        tkpt_orig[i][:, :2] = tkpt[i][:, :2] * tbox_size + tbox_min
        tkpt_orig[i][:, 2] = tkpt[i][:, 2] * area_size_xyz[2]

        skeleton.append((pkpt_orig[i], tkpt_orig[i]))

    if save_skeleton:
        for i in range(len(tgt_label_files)):
            save_file = Path(tgt_label_files[i])
            parts = list(save_file.parts)
            parts[-5] = 'predicted_label'
            save_file = Path('').joinpath(*parts)
            os.makedirs(save_file.parents[0].resolve(), exist_ok=True)
            np.save(str(save_file), skeleton[0][i])
            keypoint_error.append(np.linalg.norm(skeleton[0][i] - skeleton[1][i], axis=1))
    if len(keypoint_error) > 0:
        keypoint_error = np.stack(keypoint_error, axis=0)
    else:
        keypoint_error = np.empty([0, 21])

    return conf_matched, total_num_target, skeleton, keypoint_error


def compute_ap(conf_matched, total_num_target):
    sort_idx = np.argsort(conf_matched[:, 0])
    matched = np.flip(conf_matched[sort_idx, 1])
    matched_cum = np.cumsum(matched)
    precision = matched_cum / np.arange(1, len(matched_cum)+1)
    recall = matched_cum / total_num_target
    ap = 0
    prev_p = 0
    prev_r = 0
    for p, r in zip(precision, recall):
        if p < prev_p:
            ap += prev_p * (r - prev_r)
            prev_r = r
        prev_p = p
    return ap, (precision, recall)


def compute_pck(keypoint_error: np.ndarray, threshold_list: List[float]):
    sorted_err = np.sort(keypoint_error, axis=0).transpose()
    num_sample = sorted_err.shape[1]
    prob = np.broadcast_to(np.arange(1, num_sample+1)/num_sample, shape=sorted_err.shape)
    per_keypoint_cdf = np.stack((sorted_err, prob), axis=2)
    sorted_err = np.sort(keypoint_error, axis=None)
    num_sample = sorted_err.shape[0]
    prob = np.arange(1, num_sample+1)/num_sample
    keypoint_cdf = np.stack((sorted_err, prob), axis=1)
    per_keypoint_pck = {}
    keypoint_pck = {}
    for threshold in threshold_list:
        cond = np.float32(keypoint_error < threshold)
        num_sample = cond.shape[0]
        per_keypoint_pck[threshold] = cond.sum(axis=0)/num_sample
        num_sample = cond.size
        keypoint_pck[threshold] = cond.sum()/num_sample
    return keypoint_pck, per_keypoint_pck, keypoint_cdf, per_keypoint_cdf


def get_batch_targets(pred_boxes, target, anchors, device, ignore_thresh):
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

    target_boxes = torch.cat([torch.tensor(t.reshape(-1, 6)).float().to(pred_boxes.device) for t in target['boxes']],
                             dim=0)
    target_keypoints = torch.cat([torch.tensor(t).float().to(pred_boxes.device) for t in target['keypoints']], dim=0)
    target_keypoints = target_keypoints.reshape(-1, 63)

    target_xy = target_boxes[:, :2] * grid_size
    target_wh = target_boxes[:, 3:5] * grid_size
    t_x, t_y = target_xy.t()
    t_w, t_h = target_wh.t()

    grid_i, grid_j = target_xy.long().t()

    obj_mask[:, grid_j, grid_i] = 1
    noobj_mask[:, grid_j, grid_i] = 0

    for i in range(target_boxes.shape[0]):
        tx[i, grid_j[i], grid_i[i]] = t_x[i] - t_x[i].floor()
        ty[i, grid_j[i], grid_i][i] = t_y[i] - t_y[i].floor()

        tw[i, grid_j, grid_i] = torch.log(t_w / anchors[0] + 1e-16)
        th[i, grid_j, grid_i] = torch.log(t_h / anchors[1] + 1e-16)
        for j in range(target_keypoints.size(1)):
            tkey[i, grid_j, grid_i, j] = target_keypoints[:, j]

    output = {
        "obj_mask": obj_mask,
        "noobj_mask": noobj_mask,
        "tx": tx,
        "ty": ty,
        "tw": tw,
        "th": th,
        "tkey": tkey,
        "t_conf": obj_mask.float(),
    }

    return output