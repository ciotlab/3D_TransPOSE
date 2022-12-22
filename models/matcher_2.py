import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from einops import rearrange, repeat
import numpy as np

from util.box_ops import box_3d_cxcyczwhd_to_xyzxyz, generalized_box_3d_iou, box_iou, box_cxcywh_to_xyxy, \
    matching_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, anchor, cost_boxes: float = 1, cost_keypoint: float = 1,
                 cost_giou: float = 1, cost_obj: float = 1, iou_thresh=None):
        """Creates the matcher
        """
        super().__init__()
        self.cost_boxes = cost_boxes
        self.cost_keypoint = cost_keypoint
        self.cost_giou = cost_giou
        self.cost_obj = cost_obj
        self.iou_thresh = iou_thresh
        self.anchor = anchor

    @torch.no_grad()
    def forward(self, outputs, targets):
        # batch_size, num_queries = outputs["pred_keypoints"].shape[:2]
        batch_size, grid = outputs['pred_keypoints'].shape[:2]
        out_boxes = outputs['pred_boxes']
        x, y, z, w, h, d = [out_boxes[..., t] for t in range(6)]
        px = x + torch.arange(grid).repeat(grid, 1).view(1, grid, grid).to(x.device)
        py = y + torch.arange(grid).repeat(grid, 1).t().view(1, grid, grid).to(y.device)
        pw = torch.exp(w) * self.anchor[0]
        ph = torch.exp(h) * self.anchor[1]

        pbox = torch.stack((px, py, pw, ph), dim=-1)
        out_boxes = rearrange(pbox, "b g1 g2 c -> (b g1 g2) c")  # ( bs, query, 2, 2 ) -> ( bs*query, 4 )
        out_keypoints = rearrange(outputs["pred_keypoints"], "b g1 g2 c -> (b g1 g2) c")  # ( bs, query, 21, 3 ) -> ( bs*query, 63 )
        out_confidence = rearrange(outputs['pred_confidence'], "b g1 g2 c-> (b g1 g2) c")  # ( bs, query ) -> ( bs*query )

        tgt_boxes = rearrange(np.concatenate(targets["boxes"], axis=0), "bp p c -> bp (p c)")[ ..., [0, 1, 3, 4]] * grid # ( bs*num, 2, 2 ) -> ( bs*num, 4 )

        tgt_keypoints = rearrange(np.concatenate(targets["keypoints"], axis=0), "bp p c -> bp (p c)")  # ( bs*num, 21, 3 ) -> ( bs*num, 63 )
        tgt_boxes = torch.tensor(tgt_boxes).float().to(out_boxes.device)
        tgt_keypoints = torch.tensor(tgt_keypoints).float().to(out_boxes.device)
        
        # Compute costs
        cost_obj = -out_confidence
        cost_keypoint = torch.cdist(out_keypoints, tgt_keypoints, p=1)
        cost_boxes = torch.cdist(out_boxes, tgt_boxes, p=1)
        cost_giou = -matching_box_iou(box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes))[0]

        # Final cost matrix
        C = self.cost_keypoint * cost_keypoint + self.cost_giou * cost_giou + self.cost_obj * cost_obj \
            + self.cost_boxes * cost_boxes
        if self.iou_thresh:
            iou, union = matching_box_iou(box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            cost_iou = torch.zeros_like(iou)
            cost_iou[iou < self.iou_thresh] = 1.e+10
            C = C + cost_iou
        C = C.view(batch_size, grid * grid, -1).cpu()

        sizes = [v.shape[0] for v in targets["boxes"]]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            cm = c[i].cpu().numpy()
            index = linear_sum_assignment(cm)
            selected_rows = cm[index] < 1.e+9
            indices.append((index[0][selected_rows], index[1][selected_rows]))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
