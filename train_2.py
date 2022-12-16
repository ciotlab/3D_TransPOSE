import argparse
from pathlib import Path
import time
import json
import datetime
import os
import sys
import wandb
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import random

import util.misc as utils
from engine_2 import train_one_epoch, evaluate
from models.rddetr_2 import RDDETR_2
from models.criterion_2 import SetCriterion_2
from data_processing.dataset import get_dataset_and_dataloader_all, get_dataset_and_dataloader_num_person
import logging


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_dataset_workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=24, type=int)
    parser.add_argument('--lr_drop', default=12, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--num_stacked_seqs', default=1, type=int)
    parser.add_argument('--anchor', default=3, type=int)
    parser.add_argument('--grid_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)

    parser.add_argument('--loss_boxes_coef', default=10, type=float)
    parser.add_argument('--loss_keypoints_coef', default=1, type=float)
    parser.add_argument('--loss_iou_coef', default=25, type=float)
    parser.add_argument('--loss_object_coef', default=3, type=float)
    parser.add_argument('--empty_weight', default=60, type=float)

    parser.add_argument('--val_keypoint_thresh_list', default="0.05, 0.1, 0.2, 0.3, 0.4, 0.5", type=str)
    parser.add_argument('--val_nms_iou_thresh', default=0.5, type=float)
    parser.add_argument('--val_matching_iou_thresh', default=0.5, type=float)
    parser.add_argument('--val_conf_thresh', default=0.8, type=float)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--model_file_name', default='saved_model.pt', type=str)
    parser.add_argument('--wandb', default=False, type=bool)
    parser.add_argument('--wandb_keypoint_metric_weight', default=1, type=float)
    parser.add_argument('--wandb_keypoint_metric_thresh', default=0.2, type=float)
    return parser


def main(args):
    if args.wandb:
        wandb.init(project='3D_TransPOSE_CNN', config=args)
        args = wandb.config

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    anchor = torch.tensor([args.anchor, args.anchor]).to(device)

    model = RDDETR_2(anchor, device).to(device)
    criterion = SetCriterion_2(anchor, args.empty_weight, device, args.threshold).to(device)
    weight_dict = {'loss_keypoints': args.loss_keypoints_coef, 'loss_iou': args.loss_iou_coef,
                   'loss_boxes': args.loss_boxes_coef, 'loss_conf': args.loss_object_coef}

    if args.wandb:
        wandb.watch(model)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}, ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    train_dataloader, train_dataset = get_dataset_and_dataloader_all(args.batch_size, args.num_dataset_workers,
                                                                     args.num_stacked_seqs, mode='train')
    test_dataloader, test_dataset = get_dataset_and_dataloader_all(args.batch_size, args.num_dataset_workers,
                                                                   args.num_stacked_seqs, mode='test')
    # train_dataloader, train_dataset = \
    #     get_dataset_and_dataloader_num_person(location='B', num_person=1, test_session=1,
    #                                           batch_size=args.batch_size, num_workers=args.num_dataset_workers,
    #                                           num_stacked_seqs=args.num_stacked_seqs, mode='train')
    # test_dataloader, test_dataset = \
    #     get_dataset_and_dataloader_num_person(location='B', num_person=1, test_session=1,
    #                                           batch_size=args.batch_size, num_workers=args.num_dataset_workers,
    #                                           num_stacked_seqs=args.num_stacked_seqs, mode='test')
    logging.info("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=train_dataloader,
                                      optimizer=optimizer, device=device, epoch=epoch, weight_dict=weight_dict,
                                      max_norm=args.clip_max_norm, use_wandb=args.wandb)
        lr_scheduler.step()
        keypoint_thresh_list = [eval(x) for x in args.val_keypoint_thresh_list.split(',')]
        val_stats = evaluate(model=model, data_loader=test_dataloader, anchor=anchor, device=device,
                             threshold=args.threshold,
                             keypoint_thresh_list=keypoint_thresh_list, nms_iou_thresh=args.val_nms_iou_thresh,
                             matching_iou_thresh=args.val_matching_iou_thresh, conf_thresh=args.val_conf_thresh,
                             save_skeleton=False, save_attention_weight=False)
        ap = val_stats['AP']
        pck = val_stats['keypoint_pck']
        logging.info(f'AP: {ap}')
        logging.info(f'PCK: {pck}')
        if args.wandb:
            pck_metric = pck[args.wandb_keypoint_metric_thresh]
            combined = ap + args.wandb_keypoint_metric_weight * pck_metric
            wandb.log({'AP': ap, 'PCK': pck_metric, 'combined_metric': combined})
            pr_curve = val_stats['pr_curve']
            keypoint_cdf = val_stats['keypoint_cdf']
            pr_size = 10000
            pr_skip = int(pr_curve[0].shape[0] / pr_size)
            pr_curve = [[x, y] for (x, y) in zip(pr_curve[1][::pr_skip], pr_curve[0][::pr_skip])]
            kp_size = 1000
            kp_skip = int(keypoint_cdf.shape[0] / kp_size)
            keypoint_cdf = [[x, y] for (x, y) in keypoint_cdf[::kp_skip, :]]
            pr_curve_table = wandb.Table(data=pr_curve, columns=["recall", "precision"])
            keypoint_cdf_table = wandb.Table(data=keypoint_cdf, columns=["error_dist", "cdf"])
            wandb.log(
                {"pr_curve": wandb.plot.line(pr_curve_table, "recall", "precision", title="Precision-recall curve"),
                 "keypoint_cdf": wandb.plot.line(keypoint_cdf_table, "error_dist", "cdf", title="Keypoint error")})

    if args.wandb:
        output_file = Path(wandb.run.dir) / args.model_file_name
        torch.save(model, output_file)
    else:
        output_file = Path(__file__).parents[0].resolve() / 'saved_model' / args.model_file_name
        torch.save(model, output_file)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
