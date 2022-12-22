from typing import Iterable, Dict, List
import numpy
import numpy as np
from pathlib import Path
import os
import mpl_toolkits.mplot3d.axes3d as p3
import torch
import wandb
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from einops import rearrange, repeat

import util.misc as utils
from models.criterion_2 import get_targets
from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, box_3d_iou, iou, iou_eval, box_cxcywh_to_xyxy, box_iou, matching_box_iou
from models.matcher_2 import HungarianMatcher
from data_processing.dataset import area_min_xyz, area_size_xyz
from data_processing.dataset_visualization import tx_antenna_position, rx_antenna_position


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, weight_dict: Dict, optimizer: torch.optim.Optimizer,
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
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if use_wandb:
            wandb.log({
                       "loss": total_loss.item(),
                       "loss_boxes": loss_dict['loss_boxes'],
                       "loss_keypoints": loss_dict['loss_keypoints'],
                       "loss_conf": loss_dict['loss_conf'],
                       "loss_iou": loss_dict['loss_iou']
            })
        metric_logger.update(loss=total_loss.item(), **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, anchor, device, keypoint_thresh_list, nms_iou_thresh=0.5,
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
            = get_batch_statistics(outputs, targets, anchor, matching_iou_thresh, conf_thresh, save_skeleton,
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
    batch_size, grid_size = outputs["pred_boxes"].shape[:2]
    for b in range(batch_size):
        out_boxes = rearrange(outputs['pred_boxes'][b], "g1 g2 c -> (g1 g2) c")
        out_confidence = rearrange(outputs['pred_confidence'][b], "g1 g2 c -> (g1 g2 c)")
        sort_idx = torch.argsort(out_confidence, dim=0, descending=True).long()
        out_boxes = box_cxcywh_to_xyxy(out_boxes[sort_idx, :])
        out_confidence = out_confidence[sort_idx]
        iou, union = matching_box_iou(out_boxes, out_boxes)
        overlap = iou > iou_thresh
        for idx, conf in enumerate(out_confidence):
            if conf > 0:
                zero_idx = torch.full_like(out_confidence, False, dtype=torch.bool)
                zero_idx[idx + 1:] = overlap[idx, idx + 1:]
                out_confidence[zero_idx] = 0
        outputs['pred_confidence'].view(batch_size, grid_size * grid_size, 1)[b, sort_idx, 0] = out_confidence
    return outputs


def get_batch_statistics(outputs, targets, anchor, iou_thresh, conf_thresh, save_skeleton=False,
                         save_attention_weight=False):
    matcher = HungarianMatcher(cost_boxes=0, cost_keypoint=0, cost_giou=0, cost_obj=1,
                               iou_thresh=iou_thresh, anchor= anchor)
    indices = matcher(outputs, targets)

    bs, gs = outputs['pred_boxes'].shape[:2]
    pred_confidence = outputs['pred_confidence'].view(bs, gs*gs, 1)  # (bs, 64, 1)

    out_boxes = outputs['pred_boxes']
    x, y, z, w, h, d = [out_boxes[..., t] for t in range(6)]
    px = x + torch.arange(gs).repeat(gs, 1).view(1, gs, gs).to(x.device)
    py = y + torch.arange(gs).repeat(gs, 1).t().view(1, gs, gs).to(y.device)
    pw = torch.exp(w) * anchor[0]
    ph = torch.exp(h) * anchor[1]

    pbox = torch.stack((px/8, py/8, z, pw/8, ph/8, d), dim=-1)

    pred_boxes = pbox.view(bs, gs*gs, 2, 3)  # (bs, 64, 2, 3)
    pred_keypoints = outputs['pred_keypoints'].view(bs, gs*gs, 21, 3)

    tgt_boxes = targets['boxes']
    tgt_keypoints = targets['keypoints']
    tgt_label_files = targets['label_files']

    # extract data for AP computation
    conf_matched = []
    total_num_target = 0

    for i, index in enumerate(indices):

        pred_boxes1 = pred_boxes[i][index[0]]
        tgt_boxes1 = tgt_boxes[i]

        conf = pred_confidence[i]
        matched = torch.zeros_like(conf)
        matched[index[0]] = 1
        conf_matched.append(torch.concat((conf, matched), 1))
        num_target = tgt_boxes[i].shape[0]
        total_num_target += num_target
    conf_matched = torch.concat(conf_matched, 0)

    # extract skeleton and compute keypoint errors
    skeleton = []
    keypoint_error = []
    for i, index in enumerate(indices):
        conf = pred_confidence[i, :, 0].cpu().numpy()
        valid = conf > conf_thresh
        pbox = pred_boxes[i, valid, :, :].cpu().numpy()
        pkpt = pred_keypoints[i, valid, :, :].cpu().numpy()
        tbox = tgt_boxes[i]
        tkpt = tgt_keypoints[i]
        pbox_center = pbox[:, 0:1, :] * area_size_xyz + area_min_xyz
        pbox_size = pbox[:, 1:2, :] * area_size_xyz
        pbox_min = pbox_center - pbox_size / 2
        pkpt_orig = pkpt * pbox_size + pbox_min
        tbox_center = tbox[:, 0:1, :] * area_size_xyz + area_min_xyz
        tbox_size = tbox[:, 1:2, :] * area_size_xyz
        tbox_min = tbox_center - tbox_size / 2
        tkpt_orig = tkpt * tbox_size + tbox_min

        skeleton.append((pkpt_orig, tkpt_orig))

        if save_skeleton:
            save_file = Path(tgt_label_files[i])
            parts = list(save_file.parts)
            parts[-5] = 'predicted_label'
            save_file = Path('').joinpath(*parts)
            kpt = rearrange(pkpt_orig, 'pr kp c -> (pr kp) c')
            os.makedirs(save_file.parents[0].resolve(), exist_ok=True)
            np.save(str(save_file), kpt)
        valid_ind = np.arange(conf.shape[0])[valid]
        for p_idx, t_idx in zip(index[0].cpu().numpy(), index[1].cpu().numpy()):
            p_vidx = np.where(valid_ind == p_idx)[0]
            if p_vidx.size == 1:
                keypoint_error.append(np.linalg.norm(pkpt_orig[p_vidx[0]] - tkpt_orig[t_idx], axis=1))
    if len(keypoint_error) > 0:
        keypoint_error = np.stack(keypoint_error, axis=0)
    else:
        keypoint_error = np.empty([0, 21])
    if save_attention_weight:
        num_ant_pairs = len(tx_antenna_position) * len(rx_antenna_position)
        attention_layers = outputs['attention_map']
        for i, index in enumerate(indices):
            save_file = Path(tgt_label_files[i])
            parts = list(save_file.parts)
            parts[-5] = 'attention_weight'
            parts[-1] = parts[-1].split('.')[0] + '.png'
            save_file = Path('').joinpath(*parts)
            os.makedirs(save_file.parents[0].resolve(), exist_ok=True)
            if save_file.is_file():
                save_file.unlink()
            num_target = index[0].shape[0]
            if num_target == 0:
                break
            num_layer = len(attention_layers)
            fig = plt.figure(figsize=(4 * num_layer, 5 * num_target))
            gs = GridSpec(num_target, num_layer)
            for layer_idx, attention in enumerate(attention_layers):
                attention = rearrange(attention[i, index[0], :], 't (a s) -> t a s', a=num_ant_pairs)
                for target_idx, attention_map in enumerate(attention):
                    ax = fig.add_subplot(gs[target_idx, layer_idx])
                    ax.set_title(f'Target: {target_idx}, Layer: {layer_idx + 1}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Antenna pairs')
                    ax.imshow(attention_map.cpu().numpy())
            fig.savefig(save_file)
            plt.close(fig)
    return conf_matched, total_num_target, skeleton, keypoint_error


def compute_ap(conf_matched, total_num_target):
    sort_idx = np.argsort(conf_matched[:, 0])
    matched = np.flip(conf_matched[sort_idx, 1])
    matched_cum = np.cumsum(matched)
    matched_cum_max = matched_cum[-1]
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
