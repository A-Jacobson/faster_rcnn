import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16_bn, resnet34

from anchors import generate_anchors
from utils import (bbox_transform_inv, bbox_transform,
                   get_overlaps, filter_cross_boundary_boxes, non_max_suppression)


class BaseCNN(nn.Module):
    def __init__(self, architecture='resnet', finetune=True):
        super(BaseCNN, self).__init__()
        if architecture is 'vgg':
            vgg_full = vgg16_bn(pretrained=True)
            self.features = nn.Sequential(*list(vgg_full.features.children())[:-1])
        else:
            resnet = resnet34(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.freeze()
        if finetune:
            self.finetune()

    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def finetune(self):
        self.freeze()
        for param in self.features[-1].parameters():  # finetune only last layer
            param.requires_grad = True

    def forward(self, x):
        features = self.features(x)
        return features


class RegionProposalNetwork(nn.Module):
    def __init__(self, feature_architecture='resnet', anchor_scales=(128, 256, 512), feat_stride=16,
                 negative_overlap=0.3, positive_overlap=0.7,
                 fg_fraction=0.5, batch_size=128,
                 nms_thresh=0.7, pre_nms_limit=6000,
                 post_nms_limit=2000):
        super(RegionProposalNetwork, self).__init__()
        # Setup
        if feature_architecture == 'vgg16':
            input_dims = 512
        else:
            input_dims = 256

        self.test = False
        self.anchors = generate_anchors(feat_stride=feat_stride, scales=anchor_scales)
        self.num_anchors = self.anchors.shape[0]
        self.feat_stride = feat_stride  # how much smaller is the feature map than the original image
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap
        self.fg_fraction = fg_fraction
        self.batch_size = batch_size

        # used for both train and test
        self.nms_thresh = nms_thresh
        self.pre_nms_limit = pre_nms_limit
        self.post_nms_limit = post_nms_limit

        # for calcing targets
        self.all_anchor_boxes = None
        self.feature_map_dim = None  # (N, C, H, W)

        # layers
        self.rpn_conv1 = nn.Conv2d(input_dims, 512, kernel_size=3, padding=1)
        self.conv_classify = nn.Conv2d(512, 2 * 9, kernel_size=1)
        self.conv_bbox_regr = nn.Conv2d(512, 4 * 9, kernel_size=1)

    def forward(self, image_features):
        self.feature_map_dim = image_features.size()
        self.all_anchor_boxes = self.get_anchor_boxes(image_features)
        conv1 = F.relu(self.rpn_conv1(image_features), True)
        rpn_cls_probs = self.conv_classify(conv1)  # (N, C, H, W)
        rpn_bbox_deltas = self.conv_bbox_regr(conv1)  # (N, C, H, W)

        # https://github.com/pytorch/pytorch/issues/2013
        rpn_cls_probs = rpn_cls_probs.permute(0, 2, 3, 1).contiguous().view(-1, 2)  # reshape to cls_probs (N, 2)
        rpn_bbox_deltas = rpn_bbox_deltas.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # reshape to (N, 4)
        return rpn_cls_probs, rpn_bbox_deltas

    def get_anchor_boxes(self, image_features):
        """
        Generate anchor boxes centered at each pixel in an image feature map
        :param image_features:
        :return: K*A, 4 anchor boxes where
         K is the number of pixels in the feature map
         A is the number of anchor scales
        """
        height, width = image_features.size()[-2:]
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.shape[0]
        all_anchors = (self.anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    def get_anchor_box_labels(self, overlaps):
        """
        assigns an object label to each anchor box based on it's iou with any target
        1: positive - iou >= 0.7 or highest iou
        0: negative - iou <= 0.3
        -1: ignore - 0.3 < iou < 0.7 - will not be used to train RPN
        """
        max_overlaps = overlaps.max(axis=1)
        labels = np.zeros(len(max_overlaps))
        labels.fill(-1)  # keep track of boxes to ignore
        labels[np.where(max_overlaps <= 0.3)[0]] = 0
        labels[np.argmax(max_overlaps, axis=0)] = 1
        labels[np.where(max_overlaps >= 0.7)[0]] = 1
        return labels

    def sample_batch(self, labels, anchors, overlaps, batch_size=256):
        """
        Samples a batch of anchors from all proposed anchors.
        tries to sample a 1:1 ratio of positive to negative anchors.
        If there are too few positive anchors, pads with negatives.
        returns sampled labels, targets
        """
        num_foreground = int(0.5 * batch_size)
        foreground_indices = np.where(labels == 1)[0]

        if len(foreground_indices) > num_foreground:
            foreground_indices = np.random.choice(foreground_indices, size=num_foreground, replace=False)

        num_background = batch_size - len(foreground_indices)
        background_indices = np.where(labels == 0)[0]
        if len(background_indices) > num_background:
            background_indices = np.random.choice(background_indices, size=num_background, replace=False)
        keep = np.concatenate([foreground_indices, background_indices])
        return labels[keep], anchors[keep], overlaps[keep], keep

    def get_rpn_targets(self, targets):
        """
        :param targets: (N, x1, y1, x2, y1, C) targets
        :return: rpn_labels (batch_size, 1), rpn_bbox_targets (batch_size, 4), keep (batch_size)
        indices at which the batch was sampled)
        """
        all_anchor_boxes = self.all_anchor_boxes
        # anchor_boxes = self.filter_anchor_boxes(all_anchor_boxes)
        overlaps = get_overlaps(all_anchor_boxes, targets)
        labels = self.get_anchor_box_labels(overlaps)
        batch_labels, batch_anchor_boxes, batch_overlaps, keep = self.sample_batch(labels, all_anchor_boxes, overlaps)

        # assign anchors to targets
        anchor_assignments = np.argmax(batch_overlaps, axis=1)

        # compute target bbox deltas for rpn regressor head (256, 4)
        bbox_targets = bbox_transform(batch_anchor_boxes, targets[anchor_assignments])
        bbox_targets = torch.from_numpy(bbox_targets)
        rpn_labels = torch.from_numpy(batch_labels).long()  # convert properly
        batch_indices = torch.from_numpy(keep).long()
        return rpn_labels, bbox_targets, batch_indices  # rpn indices to keep for loss

    def get_proposal_boxes(self, rpn_bbox_deltas, rpn_cls_probs):
        """
        applies rpn bbox deltas to anchor boxes to get region proposals.
        Also filter by non-max suppression and limit to 2k boxes
        Arguments:
            rpn_bbox_deltas (Tensor) : (9*fH*fW, 4)
            rpn_cls_probs (Tensor) : (9*fH*fW,, 2)
        Return:
            proposals_boxes (Ndarray) : ( # proposal boxes, 4)
            scores (Ndarray) :  ( # proposal boxes, )
        """
        all_anchor_boxes = self.all_anchor_boxes
        nms_thresh = self.nms_thresh  # prob thresh
        pre_nms_limit = self.pre_nms_limit
        post_nms_limit = self.post_nms_limit  # eval with different numbers at test

        rpn_bbox_deltas = rpn_bbox_deltas.data.cpu().numpy()
        pos_score = rpn_cls_probs.data.cpu().numpy()[:, 1]

        # 1. Convert anchors into proposal via bbox transformation
        proposal_boxes = bbox_transform_inv(all_anchor_boxes,
                                            rpn_bbox_deltas)  # (H/16 * W/16 * 9, 4) all proposal boxes

        height, width = self.feature_map_dim[-2:]
        # 2. ignore out of bounds proposals during training
        if not self.test:
            indices = filter_cross_boundary_boxes(proposal_boxes, (height * 16, width * 16))
            proposal_boxes = proposal_boxes[indices]
            pos_score = pos_score[indices]

        # if no boxes are in the image boundaries, skip
        if len(proposal_boxes) == 0:
            return [], []
        # 3. pre nms limit
        limit = np.argsort(pos_score)[:pre_nms_limit]
        proposal_boxes = proposal_boxes[limit]
        pos_score = pos_score[limit]
        # 3. apply nms (e.g. threshold = 0.7)
        proposal_boxes, scores = non_max_suppression(proposal_boxes, pos_score, nms_thresh, post_nms_limit)
        return proposal_boxes, scores


class ROIPooling(nn.Module):
    def __init__(self, size=(7, 7), spatial_scale=16.):
        super(ROIPooling, self).__init__()
        self.adaptivepool = nn.AdaptiveMaxPool2d(size)
        self.spatial_scale = spatial_scale

    def forward(self, img_features, proposal_boxes):
        proposal_boxes = proposal_boxes.copy()  # so sneaky roi pooling doesnt scale original proposals
        proposal_boxes /= self.spatial_scale  # scale anchors to feature map size
        proposal_boxes = proposal_boxes.astype('int')
        output = []
        for bbox in proposal_boxes:
            img_crop = img_features[:, :, bbox[1]:(bbox[3] + 1), bbox[0]:(bbox[2] + 1)]
            img_warped = self.adaptivepool(img_crop)
            output.append(img_warped)
        return torch.cat(output, 0)


class Detector(nn.Module):
    def __init__(self, feature_architecture='resnet', batch_size=128, foreground_fraction=0.25,
                 foreground_threshold=0.5, background_threshold=0.5,
                 num_classes=21):
        super(Detector, self).__init__()
        # setup
        if feature_architecture == 'vgg16':
            input_dims = 512
        else:
            input_dims = 256
        self.feature_map_dim = None
        self.test = False
        self.batch_size = batch_size
        self.foreground_fraction = foreground_fraction
        self.foreground_threshold = foreground_threshold
        self.background_threshold = background_threshold
        self.num_classes = num_classes
        self.test = False
        self.num_images = 1
        self.num_foregound_proposals = None
        self.num_background_proposals = None
        self.roi_pool = ROIPooling()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_dims * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out_class = nn.Linear(4096, num_classes)
        self.out_regr = nn.Linear(4096, 4 * num_classes)

    def forward(self, img_features, proposal_boxes):
        if len(proposal_boxes) == 0:
            return [], []
        self.feature_map_dim = img_features.size()
        roi_pools = self.roi_pool(img_features, proposal_boxes)
        x = roi_pools.view(roi_pools.size(0), -1)
        x = F.relu(self.fc1(x), True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), True)
        x = self.dropout(x)
        pred_label = self.out_class(x)
        pred_bbox_deltas = self.out_regr(x)
        pred_bbox_deltas = pred_bbox_deltas.view(-1, self.num_classes, 4)
        return pred_label, pred_bbox_deltas

    def foreground_sample(self, proposal_boxes, targets):
        """
        Samples a batch of proposal boxes from all rpn proposal boxes.
         Tries to maintain ratio specified in fg/bg ratio
        :param proposal_boxes:
        :param targets:
        :return:
        """
        num_proposals = len(proposal_boxes)
        overlaps = get_overlaps(proposal_boxes, targets)
        proposal_assignments = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)  # best iou for each target box
        foreground_indices = np.where(max_overlaps >= self.foreground_threshold)[0]
        background_indices = np.where(max_overlaps < self.background_threshold)[0]

        if self.test:
            proposals_per_img = int(num_proposals / self.num_images)
        proposals_per_img = int(self.batch_size / self.num_images)
        num_foreground_proposals = int(np.round(proposals_per_img * self.foreground_fraction))

        if len(foreground_indices) > num_foreground_proposals:
            foreground_indices = np.random.choice(foreground_indices, size=num_foreground_proposals)

        num_background_proposals = proposals_per_img - num_foreground_proposals

        if len(background_indices) > num_background_proposals:
            background_indices = np.random.choice(background_indices, size=num_background_proposals)
        batch_indices = np.append(foreground_indices, background_indices)
        targets = targets[proposal_assignments]
        targets[background_indices, -1] = 0  # set background targets to 0
        targets_batch = targets[batch_indices]
        proposal_boxes_batch = proposal_boxes[batch_indices]
        return targets_batch, proposal_boxes_batch, batch_indices

    def get_targets(self, proposal_boxes, targets):
        """
        Arguments:
            proposal_boxes (Tensor) : (# proposal boxes , 4)
            targets: (N, 5)

        Return:
            labels (Ndarray) : (256,)
            bbox_deltas[:, :-1] : (256, 4)
            batch_indices for targets that were sampled
        """
        if not self.test:
            height, width = self.feature_map_dim[2:]
            indices = filter_cross_boundary_boxes(proposal_boxes, (height * 16, width * 16))
            proposal_boxes = proposal_boxes[indices]
        targets_batch, proposals_batch, batch_indices = self.foreground_sample(proposal_boxes, targets)
        bbox_deltas = bbox_transform(proposals_batch, targets_batch)
        labels_batch = targets_batch[:, -1]
        labels_batch = torch.from_numpy(labels_batch).long()
        bbox_deltas = torch.from_numpy(bbox_deltas).float()
        batch_indices = torch.from_numpy(batch_indices).long()
        return labels_batch, bbox_deltas, batch_indices


class FasterRCNN(nn.Module):
    # TODO pass settings directly to fasterrcnn (kwargs) instead of initializing seperate models
    def __init__(self, basecnn, rpn, detector):
        super(FasterRCNN, self).__init__()
        self.basecnn = basecnn
        self.rpn = rpn
        self.detector = detector

        # config
        self.test = False

        # targets
        self.proposal_boxes = None
        self.objectness_scores = None

    def forward(self, img):
        img_features = self.basecnn(img)
        # localization
        rpn_cls_probs, rpn_bbox_deltas = self.rpn(img_features)
        self.proposal_boxes, self.objectness_scores = self.rpn.get_proposal_boxes(rpn_bbox_deltas,
                                                                                  rpn_cls_probs)

        pred_label, pred_bbox_deltas = self.detector(img_features, self.proposal_boxes)
        return rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas

    def get_predictions(self, img, ignore_background=True):
        """
        :param img:
        :return: predicted_targets: (N, x1, y1, x2, y1, C)
        """
        rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = self.forward(img)
        proposals, _ = self.get_rpn_proposals()
        _, pred_class = pred_label.max(dim=1)
        pred_class = pred_class.cpu().long()
        pred_bbox_deltas = pred_bbox_deltas.data.cpu()
        idx = torch.arange(0, len(pred_class)).long()
        pred_deltas_top_class = pred_bbox_deltas[idx, pred_class.data.long()].numpy()
        pred_boxes = bbox_transform_inv(proposals, pred_deltas_top_class)
        pred_targets = np.hstack([pred_boxes, pred_class.data.cpu().numpy().reshape(-1, 1)])
        if ignore_background:
            return pred_targets[pred_targets[:, -1] != 0]
        return pred_targets

    def test_mode(self):
        self.basecnn.eval()
        self.rpn.eval()
        self.detector.eval()
        self.test = True
        self.rpn.test = True
        self.detector.test = True

    def train_mode(self):
        self.basecnn.train()
        self.rpn.train()
        self.detector.train()
        self.test = False
        self.rpn.test = False
        self.detector.test = False

    def get_rpn_proposals(self):
        """
        applies rpn bbox deltas to anchor boxes to get region proposals.
        Also filter by non-max suppression and limit to 2k boxes
        Arguments:
            rpn_bbox_deltas (Tensor) : (9*fH*fW, 4)
            rpn_cls_probs (Tensor) : (9*fH*fW,, 2)
        Return:
            proposals_boxes (Ndarray) : ( # proposal boxes, 4)
            scores (Ndarray) :  ( # proposal boxes, )
        """
        return self.proposal_boxes, self.objectness_scores

    def get_rpn_targets(self, targets):
        """
        :param targets: (N, x1, y1, x2, y1, C) targets
        :return: rpn_labels (batch_size, 1), rpn_bbox_targets (batch_size, 4), keep (batch_size)
        indices at which the batch was sampled)
        """
        return self.rpn.get_rpn_targets(targets)

    def get_detector_targets(self, targets):
        """
        Arguments:
            proposal_boxes (Tensor) : (# proposal boxes , 4)
            targets: (N, 5)

        Return:
            labels (Ndarray) : (256,)
            bbox_deltas[:, :-1] : (256, 4)
            batch_indices for targets that were sampled
        """
        return self.detector.get_targets(self.proposal_boxes, targets)
