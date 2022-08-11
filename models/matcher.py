import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from collections import Counter
import numpy as np

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, \
    generalized_3d_box_iou, box_cxcyczwhd_to_xyzxyz, volume_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_keypoint: float = 1, cost_giou: float = 1, cost_obj: float = 1,
                 device: torch.device = 'cuda'):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_class
        self.cost_keypoint = cost_keypoint
        self.cost_giou = cost_giou
        self.cost_obj = cost_obj
        self.device = device
        self.img_idx = 0
        self.threshold_pred = []
        self.processed_target = []
        self.test = False
        self.batch_ap = 0
        self.batch_statics = []
        self.grid = self._make_grid().to(self.device)
        self.anchor = torch.tensor([10.0, 10.0, 1.0]).repeat(25, 25, 1, 1).to(self.device)
        assert cost_class != 0 or cost_keypoint != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_keypoints"].shape[:2]
        out_displacement = outputs["pred_keypoints"].flatten(0, 1) # [batch_size * num_queries, num_points]
        out_boxes = outputs['pred_boxes'].flatten(0, 1)
        out_confidence = outputs['pred_confidence'].flatten(0, 1).sigmoid() # [batch_size * num_queries, 1]

        'variation of AP'
        """
        if self.test:
            prediction = torch.cat((outputs['pred_boxes'], outputs['pred_confidence'].sigmoid()), dim=-1)
            output = self.non_max_suppression(prediction)
            batch_metrics = self.get_batch_statistics(output, targets, 0.1)
            batch_metrics1 = self.get_batch_statistics(output, targets, 0.15)
            batch_metrics2 = self.get_batch_statistics(output, targets, 0.2)
            batch_metrics3 = self.get_batch_statistics(output, targets, 0.25)
            batch_metrics4 = self.get_batch_statistics(output, targets, 0.3)
            batch_metrics5 = self.get_batch_statistics(output, targets, 0.35)
            batch_metrics6 = self.get_batch_statistics(output, targets, 0.4)
            batch_metrics7 = self.get_batch_statistics(output, targets, 0.45)
            batch_metrics8 = self.get_batch_statistics(output, targets, 0.5)
            batch_metrics9 = self.get_batch_statistics(output, targets, 0.55)
            batch_metrics10 = self.get_batch_statistics(output, targets, 0.6)
            batch_metrics11 = self.get_batch_statistics(output, targets, 0.65)
            batch_metrics12 = self.get_batch_statistics(output, targets, 0.7)
            batch_metrics13 = self.get_batch_statistics(output, targets, 0.75)
            batch_metrics14 = self.get_batch_statistics(output, targets, 0.8)
            batch_metrics15 = self.get_batch_statistics(output, targets, 0.85)
            batch_metrics16 = self.get_batch_statistics(output, targets, 0.9)
            batch_metrics17 = self.get_batch_statistics(output, targets, 0.95)
            self.batch_statics += batch_metrics
            _, _, ap = self.calculate_ap(batch_metrics, targets)
            _, _, ap1 = self.calculate_ap(batch_metrics1, targets)
            _, _, ap2 = self.calculate_ap(batch_metrics2, targets)
            _, _, ap3 = self.calculate_ap(batch_metrics3, targets)
            _, _, ap4 = self.calculate_ap(batch_metrics4, targets)
            _, _, ap5 = self.calculate_ap(batch_metrics5, targets)
            _, _, ap6 = self.calculate_ap(batch_metrics6, targets)
            _, _, ap7 = self.calculate_ap(batch_metrics7, targets)
            _, _, ap8 = self.calculate_ap(batch_metrics8, targets)
            _, _, ap9 = self.calculate_ap(batch_metrics9, targets)
            _, _, ap10 = self.calculate_ap(batch_metrics10, targets)
            _, _, ap11 = self.calculate_ap(batch_metrics11, targets)
            _, _, ap12 = self.calculate_ap(batch_metrics12, targets)
            _, _, ap13 = self.calculate_ap(batch_metrics13, targets)
            _, _, ap14 = self.calculate_ap(batch_metrics14, targets)
            _, _, ap15 = self.calculate_ap(batch_metrics15, targets)
            _, _, ap16 = self.calculate_ap(batch_metrics16, targets)
            _, _, ap17 = self.calculate_ap(batch_metrics17, targets)

            self.batch_ap = (sum(ap) / bs + sum(ap1) / bs + sum(ap2) / bs + sum(ap3) / bs + sum(ap4) / bs
                             + sum(ap5) / bs + sum(ap6) / bs + sum(ap7) / bs + sum(ap8) / bs + sum(ap9) / bs
                             + sum(ap10) / bs + sum(ap11) / bs + sum(ap12) / bs + sum(ap13) / bs + sum(ap14) / bs
                             + sum(ap15) / bs + sum(ap16) / bs + sum(ap17) / bs) / 18
        """

        tgt_3d_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_displacement = torch.cat([v["keypoints"] for v in targets])
        tgt_ids = torch.cat([v['labels'] for v in targets])

        # if self.test:
        #     self.threshold_pred = []
        #     self.processed_target = []
        #     self.thresholding(outputs['pred_confidence'].sigmoid(), outputs['pred_boxes'], targets, 0.5)
        #     ap = self.mean_average_precision(self.threshold_pred, self.processed_target, 0.5)
        #     self.ap = ap
        # self.img_idx += 1

        # Calculate target keypoint displacement
        # tgt_3d_boxes_xyz = box_cxcyczwhd_to_xyzxyz(tgt_3d_bbox)[:, :3].unsqueeze(1)
        # reshape_tgt_keypoints = tgt_keypoints.reshape(tgt_keypoints.shape[0], -1, 3)
        # dist_keypoints = (reshape_tgt_keypoints - tgt_3d_boxes_xyz) / tgt_3d_bbox[:, 3:].unsqueeze(1)
        # tgt_dist_keypoints = dist_keypoints.reshape(dist_keypoints.shape[0], -1)

        # Compute the objectness
        cost_obj = -out_confidence

        # Compute the L1 cost between keypoint displacement
        cost_keypoint = torch.cdist(out_displacement, tgt_displacement, p=1)

        # compute the L1 cost between 3d boxes
        cost_3d_bbox = torch.cdist(out_boxes, tgt_3d_bbox, p=1)

        # Compute the giou cost between 3d boxes
        cost_giou = -generalized_3d_box_iou(box_cxcyczwhd_to_xyzxyz(out_boxes), box_cxcyczwhd_to_xyzxyz(tgt_3d_bbox))
        # cost_iou = -volume_iou(box_cxcyczwhd_to_xyzxyz(output_grid_box), box_cxcyczwhd_to_xyzxyz(tgt_grid_3d_bbox))[0]

        # Final cost matrix
        C = self.cost_keypoint * cost_keypoint + self.cost_giou * cost_giou + self.cost_obj * cost_obj + self.cost_bbox * cost_3d_bbox
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @staticmethod
    def _make_grid(nx=25, ny=25, nz=1):
        xv, yv, zv = torch.meshgrid([torch.arange(nx), torch.arange(ny), torch.arange(nz)])
        return torch.stack((xv, yv, zv), 2).view((nx, ny, nz, 3))