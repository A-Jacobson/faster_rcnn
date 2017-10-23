from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class RPNLoss(nn.Module):

    def forward(self, rpn_obj_scores, rpn_bbox_deltas, rpn_obj_labels, rpn_bbox_delta_targets):

        # ignore select only positive labels
        idx = rpn_obj_labels.data.ge(0).nonzero()[:, 0]
        idx = Variable(idx, requires_grad=False)
        rpn_obj_labels = rpn_obj_labels.index_select(0, idx)
        rpn_obj_scores = rpn_obj_scores.index_select(0, idx)

        classification_loss = F.cross_entropy(rpn_obj_scores, rpn_obj_labels)
        regression_loss = F.smooth_l1_loss(rpn_bbox_deltas, rpn_bbox_delta_targets)
        return classification_loss + regression_loss


class FRCNNLoss(nn.Module):

    def forward(self, cls_preds, bbox_pred, labels, bbox_targets):
        classification_loss = F.cross_entropy(cls_preds, labels)
        regression_loss = F.smooth_l1_loss(bbox_pred, bbox_targets)
        return classification_loss + regression_loss



