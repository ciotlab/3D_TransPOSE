import math
import sys
from typing import Iterable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import torch
import wandb
from collections import Counter
import matplotlib.pyplot as plt

import util.misc as utils
from util.box_ops import box_cxcyczwhd_to_xyzxyz, volume_iou, box_xyzxyz_to_cxcyczwhd


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    criterion.matcher.test = False
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # wandb.log(loss_dict_reduced_unscaled)

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir):
    model.eval()
    criterion.eval()
    criterion.matcher.test = True
    count = 0
    len_target = 0
    batch_metrics = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    for radar_signals, targets in metric_logger.log_every(data_loader, 10, header):
        radar_signals = radar_signals.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(radar_signals)
        loss_dict = criterion(predictions, targets)
        weight_dict = criterion.weight_dict
        pred_conf = predictions["pred_confidence"].sigmoid()
        pred_boxes = predictions["pred_boxes"]
        pred_keypoints = predictions["pred_keypoints"]

        prediction = torch.cat((pred_boxes, pred_conf, pred_keypoints), dim=-1)
        output = non_max_suppression(prediction)
        batch_metric = get_batch_statistics(output, targets, 0.5, device)
        batch_metrics += batch_metric
        # ap += criterion.matcher.batch_ap
        len_target += len(torch.cat([v["boxes"] for v in targets]))

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # wandb.log(loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        count += 1
    true_positives, pred_scores, error = [np.concatenate(x, 0) for x in list(zip(*batch_metrics))]
    metrics_output = calculate_ap(true_positives, pred_scores, len_target, output_dir)
    print("AP is: ", metrics_output[2][0])
    pck = error / 18888
    print("PCK is : ", pck)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return valid_stats


def make_grid(nx=25, ny=25, nz=1):
    xv, yv, zv = torch.meshgrid([torch.arange(nx), torch.arange(ny), torch.arange(nz)])
    return torch.stack((xv, yv, zv), 2).view((nx, ny, nz, 3))


def non_max_suppression(prediction, conf_thres=0.9, nms_thres=0.4):
    prediction[..., :6] = box_cxcyczwhd_to_xyzxyz(prediction[..., :6])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        image_pred = image_pred[image_pred[:, 6] >= conf_thres]
        if not image_pred.size(0):
            continue
        score = image_pred[:, 6]
        detections = image_pred[(-score).argsort()]
        keep_boxes = []
        while detections.size(0):
            large_overlap = volume_iou(detections[0, :6].unsqueeze(0), detections[:, :6])[0] > nms_thres
            invalid = large_overlap.squeeze(0)
            weights = detections[invalid, 6]
            detections[0, :6] = (weights.unsqueeze(-1) * detections[invalid, :6]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output


def get_batch_statistics(outputs, targets, iou_threshold, device):
    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output = outputs[sample_i]
        pred_boxs = output[:, :6]
        pred_scores = output[:, 6]
        pred_keypoints = output[:, 7:]
        true_positives = np.zeros(pred_boxs.shape[0])

        annotations = targets[sample_i]['boxes']
        annotations = box_cxcyczwhd_to_xyzxyz(annotations)

        target_keypoints = targets[sample_i]['keypoints']

        pred_keypoints = pred_boxs[:, :3].unsqueeze(1) + \
                         (pred_keypoints.reshape(-1, 21, 3) * box_xyzxyz_to_cxcyczwhd(pred_boxs)[:, 3:].unsqueeze(1))
        target_keypoints = annotations[:, :3].unsqueeze(1) + \
                           (target_keypoints.reshape(-1, 21, 3) * box_xyzxyz_to_cxcyczwhd(annotations)[:, 3:].unsqueeze(
                               1))

        pred_keypoints = np.array(pred_keypoints.cpu().detach())
        target_keypoints = np.array(target_keypoints.cpu().detach())

        pred_keypoints = pred_keypoints * np.array([3, 3, 2])
        target_keypoints = target_keypoints * np.array([3, 3, 2])

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations
            for pred_i, pred_box in enumerate(pred_boxs):
                if len(detected_boxes) == len(annotations):
                    break
                error = np.linalg.norm(pred_keypoints[pred_i] - target_keypoints, ord=2, axis=-1)
                iou, box_index = volume_iou(pred_box.unsqueeze(0), target_boxes)[0].max(-1)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores.cpu().detach().numpy(), error])
    return batch_metrics


def calculate_ap(tp, conf, n_gt, output_dir):
    ap, p, r = [], [], []
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]
    # tp = tp[2:]
    # n_gt = len(torch.cat([v["boxes"] for v in targets]))
    n_p = len(tp)
    if n_p == 0 or n_gt == 0:
        ap.append(0)
        r.append(0)
        p.append(0)
    else:
        fpc = (1 - tp).cumsum()
        tpc = tp.cumsum()
        recall_curve = tpc / (n_gt + 1e-16)
        precision_curve = tpc / (tpc + fpc)
        ap.append(compute_ap(recall_curve, precision_curve))
        r.append(recall_curve[-1])
        p.append(precision_curve[-1])
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axis([0, 1, 0, 1])
        plt.savefig(output_dir + 'ap.png')
        # plt.show()
    return p, r, ap


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # plt.plot(mrec, mpre)
    # plt.show()
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


@torch.no_grad()
def test(model, criterion, data_loader, device, iou_threshold, imaging, output_dir):
    model.eval()
    criterion.eval()
    criterion.test = True
    imaging_interval = 1
    pck = 0
    # plt.colorbar()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for radar_signals, targets in metric_logger.log_every(data_loader, 10, header):
        radar_signals = radar_signals.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(radar_signals)
        loss_dict = criterion(predictions, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # wandb.log(loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        attention_map = criterion.attention_map.squeeze()
        attention_map = attention_map.cpu().detach().numpy()
        # attention_map = np.clip(attention_map, 0, 0.001)
        # plt.colorbar()
        """
        for i in range(len(attention_map)):
            if i == 5:
                save_path = './result_image/journal/testing_5/attention_map/position_encode/{}/'.format(i)
                Path(save_path).mkdir(parents=True, exist_ok=True)
                token_attention = attention_map[i]
                for k in range(len(token_attention)):
                    tokenize_attention = token_attention[k].reshape((62, 64, 1))
                    tokenize_attention = tokenize_attention.transpose(1, 0, 2)
                    plt.imshow(tokenize_attention)
                    plt.savefig(save_path + '{}.png'.format(k))
        """
        # attention_map = attention_map.reshape((64, 62, 1))
        # attention_map = attention_map.transpose(1, 0, 2)
        # plt.imshow(attention_map)
        # plt.colorbar()
        # attention_map = criterion.attention_map.view(9, 9, 6, 1)
        # plt.imshow(attention_map.cpu().detach().numpy())
        # if imaging_interval == 1:
        #     plt.colorbar()
        # if imaging_interval % 10 == 0:
        #     # plt.imshow(attention_map)
        #     plt.savefig('./result_image/journal/testing_2/attention_map/{}.png'.format(imaging_interval))

        pred_conf = predictions["pred_confidence"].sigmoid()
        pred_boxes = predictions["pred_boxes"]
        pred_keypoints = predictions["pred_keypoints"]

        prediction = torch.cat((pred_boxes, pred_conf, pred_keypoints), dim=-1)
        output = non_max_suppression(prediction)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim([0, 3])
        ax.set_ylim([0, 3])
        ax.set_zlim([0, 2])

        if imaging_interval % 10 == 0:
            for target_i in range(len(targets)):
                if targets[target_i] is None:
                    continue
                targets = targets[target_i]
                target_boxes = targets["boxes"]
                target_boxes = box_cxcyczwhd_to_xyzxyz(target_boxes)
                target_keypoints = targets["keypoints"]
                target_keypoints = target_boxes[:, :3].unsqueeze(1) + (
                            target_keypoints.reshape(-1, 21, 3) * box_xyzxyz_to_cxcyczwhd(target_boxes)[:, 3:].unsqueeze(1))
                target_keypoints = np.array(target_keypoints.cpu().detach())
                target_keypoints = np.reshape(target_keypoints, (-1, 21, 3))
                target_keypoints = target_keypoints * np.array([3, 3, 2])

                for k in range(len(target_keypoints)):
                    for i in range(len(target_keypoints[k])):
                        ax.scatter3D(target_keypoints[k][i][0], target_keypoints[k][i][1], target_keypoints[k][i][2],
                                     c='Red', s=10)
                    x = [0, 5, 9, 13, 16]
                    y = [4, 8, 12, 15, 18]
                    for n1, n2 in zip(x, y):
                        for i in range(n1, n2):
                            ax.plot([target_keypoints[k][i][0], target_keypoints[k][i + 1][0]],
                                    [target_keypoints[k][i][1], target_keypoints[k][i + 1][1]],
                                    [target_keypoints[k][i][2], target_keypoints[k][i + 1][2]], c='Red')
                    ax.plot([target_keypoints[k][3][0], target_keypoints[k][5][0]],
                            [target_keypoints[k][3][1], target_keypoints[k][5][1]],
                            [target_keypoints[k][3][2], target_keypoints[k][5][2]], c='Red')
                    ax.plot([target_keypoints[k][3][0], target_keypoints[k][9][0]],
                            [target_keypoints[k][3][1], target_keypoints[k][9][1]],
                            [target_keypoints[k][3][2], target_keypoints[k][9][2]], c='Red')
                    ax.plot([target_keypoints[k][0][0], target_keypoints[k][13][0]],
                            [target_keypoints[k][0][1], target_keypoints[k][13][1]],
                            [target_keypoints[k][0][2], target_keypoints[k][13][2]], c='Red')
                    ax.plot([target_keypoints[k][0][0], target_keypoints[k][16][0]],
                            [target_keypoints[k][0][1], target_keypoints[k][16][1]],
                            [target_keypoints[k][0][2], target_keypoints[k][16][2]], c='Red')
                    ax.plot([target_keypoints[k][15][0], target_keypoints[k][19][0]],
                            [target_keypoints[k][15][1], target_keypoints[k][19][1]],
                            [target_keypoints[k][15][2], target_keypoints[k][19][2]], c='Red')
                    ax.plot([target_keypoints[k][18][0], target_keypoints[k][20][0]],
                            [target_keypoints[k][18][1], target_keypoints[k][20][1]],
                            [target_keypoints[k][18][2], target_keypoints[k][20][2]], c='Red')

            for sample_i in range(len(output)):
                if output[sample_i] is None:
                    continue
                output = output[sample_i]
                pred_boxes = output[:, :6]
                pred_keypoints = output[:, 7:]
                pred_keypoints = pred_boxes[:, :3].unsqueeze(1) + (
                            pred_keypoints.reshape(-1, 21, 3) * box_xyzxyz_to_cxcyczwhd(pred_boxes)[:, 3:].unsqueeze(1))
                src_keypoints = np.array(pred_keypoints.cpu().detach())
                src_keypoints = np.reshape(src_keypoints, (-1, 21, 3))
                src_keypoints = src_keypoints * np.array([3, 3, 2])

                for k in range(len(src_keypoints)):
                    for i in range(len(src_keypoints[k])):
                        ax.scatter3D(src_keypoints[k][i][0], src_keypoints[k][i][1], src_keypoints[k][i][2],
                                     c='Green', s=10)
                    x = [0, 5, 9, 13, 16]
                    y = [4, 8, 12, 15, 18]
                    for n1, n2 in zip(x, y):
                        for i in range(n1, n2):
                            ax.plot([src_keypoints[k][i][0], src_keypoints[k][i + 1][0]],
                                    [src_keypoints[k][i][1], src_keypoints[k][i + 1][1]],
                                    [src_keypoints[k][i][2], src_keypoints[k][i + 1][2]], c='Green')
                    ax.plot([src_keypoints[k][3][0], src_keypoints[k][5][0]],
                            [src_keypoints[k][3][1], src_keypoints[k][5][1]],
                            [src_keypoints[k][3][2], src_keypoints[k][5][2]], c='Green')
                    ax.plot([src_keypoints[k][3][0], src_keypoints[k][9][0]],
                            [src_keypoints[k][3][1], src_keypoints[k][9][1]],
                            [src_keypoints[k][3][2], src_keypoints[k][9][2]], c='Green')
                    ax.plot([src_keypoints[k][0][0], src_keypoints[k][13][0]],
                            [src_keypoints[k][0][1], src_keypoints[k][13][1]],
                            [src_keypoints[k][0][2], src_keypoints[k][13][2]], c='Green')
                    ax.plot([src_keypoints[k][0][0], src_keypoints[k][16][0]],
                            [src_keypoints[k][0][1], src_keypoints[k][16][1]],
                            [src_keypoints[k][0][2], src_keypoints[k][16][2]], c='Green')
                    ax.plot([src_keypoints[k][15][0], src_keypoints[k][19][0]],
                            [src_keypoints[k][15][1], src_keypoints[k][19][1]],
                            [src_keypoints[k][15][2], src_keypoints[k][19][2]], c='Green')
                    ax.plot([src_keypoints[k][18][0], src_keypoints[k][20][0]],
                            [src_keypoints[k][18][1], src_keypoints[k][20][1]],
                            [src_keypoints[k][18][2], src_keypoints[k][20][2]], c='Green')
                    plt.legend('Target')
                    fig.savefig(output_dir + '{}.png'.format(imaging_interval))

        imaging_interval += 1
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


def mean_average_precision(pred, target, iou_threshold):
    epsilon = 1e-6
    pred.sort(key=lambda x: x[1], reverse=True)
    TP = torch.zeros((len(pred)))
    FP = torch.zeros((len(pred)))
    total_num_box = len(target)
    amount_bboxes = Counter(t[0] for t in target)
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)
    for i, p in enumerate(pred):
        ground_truth = [bbox for bbox in target if bbox[0] == p[0]]
        best_iou = 0

        for idx, gt in enumerate(ground_truth):
            iou = volume_iou(box_cxcyczwhd_to_xyzxyz(p[2].unsqueeze(0)), box_cxcyczwhd_to_xyzxyz(gt[1].unsqueeze(0)))[
                0].squeeze()
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            if amount_bboxes[p[0]][best_gt_idx] == 0:
                TP[i] = 1
                amount_bboxes[p[0]][best_gt_idx] = 1
            else:
                FP[i] = 1
        else:
            FP[i] = 1
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_num_box + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

    recalls = torch.cat((torch.tensor([0]), recalls))
    precisions = torch.cat((torch.tensor([1]), precisions))
    average_precisions = torch.trapz(precisions, recalls)
    return average_precisions
