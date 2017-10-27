from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class RPNLoss(nn.Module):

    def forward(self, rpn_obj_scores, rpn_bbox_deltas, rpn_obj_labels, rpn_bbox_delta_targets, keep_indices):
        # ignore select only positive labels
        classification_loss = F.cross_entropy(rpn_obj_scores[keep_indices], rpn_obj_labels)
        regression_loss = F.smooth_l1_loss(rpn_bbox_deltas[keep_indices], rpn_bbox_delta_targets)
        return classification_loss + regression_loss


class FRCNNLoss(nn.Module):

    def forward(self, pred_label, pred_bbox_deltas, labels, bbox_deltas, keep_indices):
        classification_loss = F.cross_entropy(pred_label[keep_indices], labels)
        regression_loss = F.smooth_l1_loss(pred_bbox_deltas[keep_indices, labels.data, :], bbox_deltas)
        return classification_loss + regression_loss



