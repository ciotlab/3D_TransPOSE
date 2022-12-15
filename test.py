import argparse
from pathlib import Path
import torch
import numpy as np
from engine import evaluate
from data_processing.dataset import get_dataset_and_dataloader_all, get_dataset_and_dataloader_num_person
import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_dataset_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_stacked_seqs', default=2, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--keypoint_thresh_list', default="0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75", type=str)
    parser.add_argument('--nms_iou_thresh', default=0.5, type=float)
    # parser.add_argument('--matching_iou_thresh', default=0.1, type=float)
    # parser.add_argument('--matching_iou_thresh', default=0.2, type=float)
    parser.add_argument('--matching_iou_thresh', default=0.3, type=float)
    # parser.add_argument('--matching_iou_thresh', default=0.5, type=float)
    # parser.add_argument('--matching_iou_thresh', default=0.75, type=float)

    parser.add_argument('--conf_thresh', default=0.9, type=float)
    parser.add_argument('--model_file_name', default='saved_model.pt', type=str)
    parser.add_argument('--save_skeleton', default=True, type=bool)
    parser.add_argument('--graph_file_name', default='graph.png', type=str)
    parser.add_argument('--pr_curve_file_name', default='pr_curve.txt', type=str)
    parser.add_argument('--kp_cdf_file_name', default='kp_cdf.txt', type=str)
    parser.add_argument('--save_attention_weight', default=True, type=bool)
    return parser


def main(args):
    device = torch.device(args.device)
    test_dataloader, test_dataset = get_dataset_and_dataloader_all(args.batch_size, args.num_dataset_workers,
                                                                   args.num_stacked_seqs, mode='test')
    # test_dataloader, test_dataset = \
    #     get_dataset_and_dataloader_num_person(location='B', num_person=1, test_session=1,
    #                                           batch_size=args.batch_size, num_workers=args.num_dataset_workers,
    #                                           num_stacked_seqs=args.num_stacked_seqs, mode='test')
    model_file = Path(__file__).parents[0].resolve() / 'saved_model' / args.model_file_name
    model = torch.load(model_file)
    keypoint_thresh_list = [eval(x) for x in args.keypoint_thresh_list.split(',')]
    stats = evaluate(model=model, data_loader=test_dataloader, device=device,
                     keypoint_thresh_list=keypoint_thresh_list, nms_iou_thresh=args.nms_iou_thresh,
                     matching_iou_thresh=args.matching_iou_thresh,
                     conf_thresh=args.conf_thresh, save_skeleton=args.save_skeleton,
                     save_attention_weight=args.save_attention_weight)
    ap = stats['AP']
    pck = stats['keypoint_pck']
    logging.info(f'AP: {ap}')
    logging.info(f'PCK: {pck}')
    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2)
    pr_curve = stats['pr_curve']
    recall = pr_curve[1]
    precision = pr_curve[0]
    interval = int(recall.shape[0] / 1000)
    if interval > 0:
        recall = recall[::interval]
        precision = precision[::interval]
    ax_ap = fig.add_subplot(gs[0, 0])
    ax_ap.set_title('Precision-recall curve')
    ax_ap.set_xlabel('Recall')
    ax_ap.set_ylabel('Precision')
    ax_ap.plot(recall, precision)
    keypoint_cdf = stats['keypoint_cdf']
    interval = int(keypoint_cdf.shape[0] / 1000)
    if interval > 0:
        keypoint_cdf = keypoint_cdf[::interval]
    ax_kp = fig.add_subplot(gs[0, 1])
    ax_kp.set_title('CDF of keypoint error')
    ax_kp.set_xlabel('Error (m)')
    ax_kp.set_ylabel('CDF')
    ax_kp.plot(keypoint_cdf[:, 0], keypoint_cdf[:, 1])
    output_path = Path(__file__).parents[0].resolve() / 'outputs'
    graph_file = output_path / args.graph_file_name
    fig.savefig(graph_file)
    pr_curve_file = output_path / args.pr_curve_file_name
    tmp = np.stack((recall, precision), axis=1)
    np.savetxt(pr_curve_file, tmp, delimiter=",")
    kp_cdf_file = output_path / args.kp_cdf_file_name
    np.savetxt(kp_cdf_file, keypoint_cdf, delimiter=",")


if __name__ == '__main__':
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
