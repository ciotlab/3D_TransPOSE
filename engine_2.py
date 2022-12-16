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
from einops import rearrange

import util.misc as utils
from models.criterion_2 import get_targets
from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, box_3d_iou, iou, iou_eval
from models.matcher import HungarianMatcher
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
            wandb.log({"loss": total_loss.item(),
                       "loss_boxes": loss_dict['loss_boxes'],
                       "loss_keypoints": loss_dict['loss_keypoints'],
                       "loss_conf": loss_dict['loss_conf'],
                       "loss_iou": loss_dict['loss_iou']})
        metric_logger.update(loss=total_loss.item(), **loss_dict)
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
        tp, num_target, skeleton, keypoint_error \
            = get_batch_statistics(outputs, targets, anchor, device, threshold, matching_iou_thresh, conf_thresh,
                                   save_skeleton,
                                   save_attention_weight)
        conf_matched_list = conf_matched_list + tp
        total_num_target += num_target
        skeleton_list.extend(skeleton)
        keypoint_error_list.append(keypoint_error)
    ap, pr_curve = compute_ap(conf_matched_list, total_num_target)
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
                zero_idx[idx + 1:] = overlap[idx, idx + 1:]
                out_confidence[zero_idx] = 0
        outputs['pred_confidence'][b, sort_idx, 0] = out_confidence
    return outputs


def get_batch_statistics(outputs, targets, anchor, device, threshold, iou_thresh, conf_thresh, save_skeleton=False,
                         save_attention_weight=False, ):
    batch_size = outputs.size(0)
    grid_size = outputs.size(1)
    px = outputs[:, :, :, 0]
    py = outputs[:, :, :, 1]
    pw = outputs[:, :, :, 2]
    ph = outputs[:, :, :, 3]
    pred_conf = outputs[:, :, :, 4]
    pred_keypoints = outputs[:, :, :, 5:]

    tgt_boxes = targets['boxes']
    tgt_keypoints = targets['keypoints']
    tgt_label_files = targets['label_files']
    total_num_target = 0
    concat_tp = []

    tgts = get_targets(pred_conf, targets, device)
    tx = tgts["tx"]
    ty = tgts["ty"]
    tw = tgts["tw"]
    th = tgts["th"]
    tkey = tgts["tkey"]
    t_confidence = tgts["t_conf"]
    obj_mask = tgts["obj_mask"]

    # extract skeleton and compute keypoint errors
    skeleton = []
    keypoint_error = []

    for bs in range(batch_size):
        target_boxes = torch.tensor(targets['boxes'][bs].reshape(-1, 6)).float().to(pred_conf.device)
        target_keypoints = torch.tensor(targets['keypoints'][bs].reshape(-1, 63)).float().to(pred_conf.device)
        target_xy = target_boxes[:, :2] * grid_size
        target_wh = target_boxes[:, 3:5] * grid_size

        grid_i, grid_j = target_xy.long().t()

        p_conf = pred_conf[bs][obj_mask[bs]].cpu().numpy()
        p_x = px[bs][obj_mask[bs]].cpu().numpy()
        p_y = py[bs][obj_mask[bs]].cpu().numpy()
        p_w = pw[bs][obj_mask[bs]].cpu().numpy()
        p_h = ph[bs][obj_mask[bs]].cpu().numpy()
        pkpt = pred_keypoints[bs][obj_mask[bs]].cpu().numpy().reshape(-1, 21, 3)

        t_x = tx[bs][obj_mask[bs]].cpu().numpy()
        t_y = ty[bs][obj_mask[bs]].cpu().numpy()
        t_w = tw[bs][obj_mask[bs]].cpu().numpy()
        t_h = th[bs][obj_mask[bs]].cpu().numpy()
        tkpt = tkey[bs][obj_mask[bs]].cpu().numpy().reshape(-1, 21, 3)

        pbox = np.stack((p_x, p_y, p_w, p_h)).T
        tbox = np.stack((t_x, t_y, t_w, t_h)).T

        pbox_center = pbox[:, :2] * area_size_xyz[:2] + area_min_xyz[:2]
        pbox_size = pbox[:, 2:] * area_size_xyz[:2]
        pbox_min = pbox_center - pbox_size / 2
        pkpt_orig = np.zeros(pkpt.shape)

        tbox_center = tbox[:, :2] * area_size_xyz[:2] + area_min_xyz[:2]
        tbox_size = tbox[:, 2:] * area_size_xyz[:2]
        tbox_min = tbox_center - tbox_size / 2
        tkpt_orig = np.zeros(tkpt.shape)

        for i in range(tkpt.shape[0]):

            # count number of target
            total_num_target += 1

            p_x[i] += grid_i[i].cpu().numpy()
            p_y[i] += grid_j[i].cpu().numpy()
            p_w[i] = numpy.exp(p_w[i]) * anchor[0].cpu().numpy()
            p_h[i] = numpy.exp(p_h[i]) * anchor[1].cpu().numpy()

            pkpt_orig[i][:, :2] = pkpt[i][:, :2] * pbox_size[i] + pbox_min[i]
            pkpt_orig[i][:, 2] = pkpt[i][:, 2] * area_size_xyz[2]

            tkpt_orig[i][:, :2] = tkpt[i][:, :2] * tbox_size[i] + tbox_min[i]
            tkpt_orig[i][:, 2] = tkpt[i][:, 2] * area_size_xyz[2]

            box_iou = iou_eval(p_x[i], p_y[i], p_w[i], p_h[i], t_x[i], t_y[i], t_w[i], t_h[i])
            # print('box_iou : ', box_iou, " p_conf : ", p_conf[i])

            if (box_iou >= iou_thresh) & (p_conf[i] >= conf_thresh):
                concat_tp.append(1)
            else:
                concat_tp.append(0)

        skeleton.append((pkpt_orig, tkpt_orig))

        if save_skeleton:
            save_file = Path(tgt_label_files[bs])
            parts = list(save_file.parts)
            parts[-5] = 'predicted_label'
            save_file = Path('').joinpath(*parts)
            kpt = rearrange(pkpt_orig, 'pr kp c -> (pr kp) c')
            os.makedirs(save_file.parents[0].resolve(), exist_ok=True)
            np.save(str(save_file), kpt)

        for i in range(tkpt.shape[0]):
            keypoint_error.append(np.linalg.norm(pkpt_orig[i] - tkpt_orig[i], axis=1))
    if len(keypoint_error) > 0:
        keypoint_error = np.stack(keypoint_error, axis=0)
    else:
        keypoint_error = np.empty([0, 21])

    return concat_tp, total_num_target, skeleton, keypoint_error


def compute_ap(conf_matched, total_num_target):
    matched_cum = np.cumsum(conf_matched)
    precision = matched_cum / np.arange(1, len(matched_cum) + 1)
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
    prob = np.broadcast_to(np.arange(1, num_sample + 1) / num_sample, shape=sorted_err.shape)
    per_keypoint_cdf = np.stack((sorted_err, prob), axis=2)
    sorted_err = np.sort(keypoint_error, axis=None)
    num_sample = sorted_err.shape[0]
    prob = np.arange(1, num_sample + 1) / num_sample
    keypoint_cdf = np.stack((sorted_err, prob), axis=1)
    per_keypoint_pck = {}
    keypoint_pck = {}
    for threshold in threshold_list:
        cond = np.float32(keypoint_error < threshold)
        num_sample = cond.shape[0]
        per_keypoint_pck[threshold] = cond.sum(axis=0) / num_sample
        num_sample = cond.size
        keypoint_pck[threshold] = cond.sum() / num_sample
    return keypoint_pck, per_keypoint_pck, keypoint_cdf, per_keypoint_cdf
