import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from einops import rearrange
import numpy as np

from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, generalized_box_3d_iou, box_3d_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_boxes: float = 1, cost_keypoint: float = 1,
                 cost_giou: float = 1, cost_obj: float = 1, iou_thresh=None):
        """Creates the matcher
        """
        super().__init__()
        self.cost_boxes = cost_boxes
        self.cost_keypoint = cost_keypoint
        self.cost_giou = cost_giou
        self.cost_obj = cost_obj
        self.iou_thresh = iou_thresh

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["pred_keypoints"].shape[:2]
        out_boxes = rearrange(outputs['pred_boxes'], "b q p c -> (b q) (p c)")
        out_keypoints = rearrange(outputs["pred_keypoints"], "b q p c -> (b q) (p c)")
        out_confidence = rearrange(outputs['pred_confidence'], "b q c -> (b q) c")
        tgt_boxes = rearrange(np.concatenate(targets["boxes"], axis=0), "bp p c -> bp (p c)")
        tgt_keypoints = rearrange(np.concatenate(targets["keypoints"], axis=0), "bp p c -> bp (p c)")
        tgt_boxes = torch.tensor(tgt_boxes).float().to(out_boxes.device)
        tgt_keypoints = torch.tensor(tgt_keypoints).float().to(out_boxes.device)
        # Compute costs
        cost_obj = -out_confidence
        cost_keypoint = torch.cdist(out_keypoints, tgt_keypoints, p=1)
        cost_boxes = torch.cdist(out_boxes, tgt_boxes, p=1)
        cost_giou = -generalized_box_3d_iou(box_3d_cxcyczwhd_to_xyzxyz(out_boxes), box_3d_cxcyczwhd_to_xyzxyz(tgt_boxes))

        # Final cost matrix
        C = self.cost_keypoint * cost_keypoint + self.cost_giou * cost_giou + self.cost_obj * cost_obj \
            + self.cost_boxes * cost_boxes
        if self.iou_thresh:
            iou, union = box_3d_iou(box_3d_cxcyczwhd_to_xyzxyz(out_boxes), box_3d_cxcyczwhd_to_xyzxyz(tgt_boxes))
            cost_iou = torch.zeros_like(iou)
            cost_iou[iou < self.iou_thresh] = 1.e+10
            C = C + cost_iou
        C = C.view(batch_size, num_queries, -1).cpu()

        sizes = [v.shape[0] for v in targets["boxes"]]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            cm = c[i].cpu().numpy()
            index = linear_sum_assignment(cm)
            selected_rows = cm[index] < 1.e+9
            indices.append((index[0][selected_rows], index[1][selected_rows]))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
