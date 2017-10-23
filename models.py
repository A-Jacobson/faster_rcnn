import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16_bn

from bbox_utils import (bbox_transform_inv, bbox_overlaps,
                        bbox_transform, unmap, clip_boxes, filter_boxes, non_max_suppression)
from generate_anchors import generate_anchors
from utils import boxes_to_numpy, get_output_dim, classes_to_numpy


class Base_CNN(nn.Module):
    # TODO: ADD RESNET OPTION
    def __init__(self):
        super(Base_CNN, self).__init__()
        vgg_full = vgg16_bn(pretrained=True)
        for param in vgg_full.parameters():
            param.requires_grad = False
        self.vggfeats = nn.Sequential(*list(vgg_full.features.children())[:-1])  # drop last pooling layer

    def forward(self, x):
        return self.vggfeats(x)


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_scales=None, feat_stride=16,
                 negative_overlap=0.3, positive_overlap=0.7,
                 fg_fraction=0.5, batch_size=256,
                 nms_thresh=0.7, min_size=16,
                 pre_nms_topN=12000, post_nms_topN=2000
                 ):
        super(RegionProposalNetwork, self).__init__()
        # layers
        self.rpn_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_classify = nn.Conv2d(512, 2 * 9, kernel_size=1)
        self.conv_bbox_regr = nn.Conv2d(512, 4 * 9, kernel_size=1)

        # Setup
        if anchor_scales is None:
            anchor_scales = (8, 16, 32)
        self._anchors = generate_anchors(base_size=16, scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self.feat_stride = feat_stride
        self.negative_overlap = negative_overlap
        self.positive_overlap = positive_overlap
        self.fg_fraction = fg_fraction
        self.batch_size = batch_size

        # used for both train and test
        self.nms_thresh = nms_thresh
        self.pre_nms_topN = pre_nms_topN
        self.post_nms_topN = post_nms_topN
        self.min_size = min_size

        # for calcing targets
        self.all_anchor_boxes = None

    def forward(self, image_features):
        self.all_anchor_boxes = self.rpn_get_anchors(image_features)
        conv1 = F.relu(self.rpn_conv1(image_features), True)
        rpn_cls_probs = self.conv_classify(conv1)  # (N, C, H, W)
        rpn_bbox_preds = self.conv_bbox_regr(conv1)  # (N, C, H, W)

        # https://github.com/pytorch/pytorch/issues/2013
        rpn_cls_probs = rpn_cls_probs.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        return rpn_cls_probs, rpn_bbox_preds

    def rpn_get_anchors(self, image_features):
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
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    def get_rpn_targets(self, targets, img):
        """
        Arguments:
            all_anchors_boxes (Tensor) : (H/16 * W/16 * 9, 4)
            gt_boxes_c (Ndarray) : (# gt boxes, 5) [x, y, x`, y`, class]
            im_info (Tuple) : (Height, Width, Channel, Scale)
            args (argparse.Namespace) : global arguments

        Return:
            labels (Ndarray) : (H/16 * W/16 * 9,)
            bbox_targets (Ndarray) : (H/16 * W/16 * 9, 4)
            bbox_inside_weights (Ndarray) : (H/16 * W/16 * 9, 4)
        """
        all_anchor_boxes = self.all_anchor_boxes
        rpn_batch_size = 256
        # it maybe H/16 * W/16 * 9
        num_anchors = all_anchor_boxes.shape[0]
        height, width = img.size()[-2:]

        # only keep anchors inside the image
        _allowed_border = 0
        inds_inside = np.where(
            (all_anchor_boxes[:, 0] >= -_allowed_border) &
            (all_anchor_boxes[:, 1] >= -_allowed_border) &
            (all_anchor_boxes[:, 2] < width + _allowed_border) &  # width
            (all_anchor_boxes[:, 3] < height + _allowed_border)  # height
        )[0]

        # keep only inside anchors
        inside_anchors_boxes = all_anchor_boxes[inds_inside, :]
        assert inside_anchors_boxes.shape[0] > 0, '{0}x{1} -> {2}'.format(height, width, num_anchors)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        target_bbs = boxes_to_numpy(targets)

        # iou
        overlaps = bbox_overlaps(inside_anchors_boxes, target_bbs).cpu().numpy()

        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

        gt_argmax_overlaps = overlaps.argmax(axis=0)

        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < 0.3] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= 0.7] = 1

        # subsample positive labels if we have too many
        num_fg = int(0.5 * rpn_batch_size)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = 256 - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        # transform boxes to deltas boxes
        bbox_targets = bbox_transform(inside_anchors_boxes, target_bbs[argmax_overlaps])

        # maybe use positive mask for training?

        # map up to original set of anchors
        feature_height, feature_width = get_output_dim(img)

        # convert for L1 loss
        bbox_targets = unmap(bbox_targets, num_anchors, inds_inside, fill=0)
        bbox_targets = torch.from_numpy(bbox_targets)
        bbox_targets = bbox_targets.view(1, feature_height, feature_width, -1).permute(0, 3, 1, 2)

        # convert for cross entropy loss
        labels = unmap(labels, num_anchors, inds_inside, fill=-1)
        labels = torch.from_numpy(labels).long()  # convert properly
        labels = labels.view(1, feature_height, feature_width, -1).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(-1)

        return labels, bbox_targets

    def get_roi_boxes(self, rpn_bbox_pred, rpn_cls_prob, img, target, test):
        """
        applies rpn bbox deltas to anchor boxes to get region proposals. Also filters by various means
        including objectness score

        Arguments:
            rpn_bbox_pred (Tensor) : (1, 4*9, H/16, W/16)
            rpn_cls_prob (Tensor) : (1, 2*9, H/16, W/16)
            all_anchors_boxes (Ndarray) : (H/16 * W/16 * 9, 4) predicted boxes
            im_info (Tuple) : (Height, Width, Channel, Scale)
            test (Bool) : True or False
            args (argparse.Namespace) : global arguments

        Return:

            # in each minibatch number of proposal boxes is variable
            proposals_boxes (Ndarray) : ( # proposal boxes, 4)
            scores (Ndarray) :  ( # proposal boxes, )
        """
        all_anchor_boxes = self.all_anchor_boxes
        # if test == False, using training args else using testing args
        pre_nms_topn = 12000 if not test else 6000  # filter by topn prob
        nms_thresh = 0.7 if not test else 0.7  # prob thresh
        post_nms_topn = 2000 if not test else 300

        bbox_deltas = rpn_bbox_pred.data.cpu().numpy()  # how much to change proposed bboxes?

        # 1. Convert anchors into proposal via bbox transformation
        proposals_boxes = bbox_transform_inv(all_anchor_boxes, bbox_deltas)  # (H/16 * W/16 * 9, 4) all proposal boxes
        pos_score = rpn_cls_prob.data.cpu()[:, 1].unsqueeze(-1).numpy()

        height, width = img.size()[-2:]

        # if test==True, keep anchors inside the image
        # if test==False, delete anchors inside the image
        if not test:
            _allowed_border = 0
            inds_inside = np.where(
                (all_anchor_boxes[:, 0] >= -_allowed_border) &
                (all_anchor_boxes[:, 1] >= -_allowed_border) &
                (all_anchor_boxes[:, 2] < width + _allowed_border) &  # width
                (all_anchor_boxes[:, 3] < height + _allowed_border)  # height
            )[0]

            mask = np.zeros(proposals_boxes.shape[0], dtype=bool)
            mask[inds_inside] = True

            proposals_boxes = proposals_boxes[mask]
            pos_score = pos_score[mask]

        # 2. clip proposal boxes to image
        proposals_boxes = clip_boxes(proposals_boxes, (height, width))

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[3])
        filter_indices = filter_boxes(proposals_boxes, 10 * max(target[0]['scale'][0]))

        # delete filter_indices
        mask = np.zeros(proposals_boxes.shape[0], dtype=bool)
        mask[filter_indices] = True

        proposals_boxes = proposals_boxes[mask]
        pos_score = pos_score[mask]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        indices = np.argsort(pos_score.squeeze())[::-1]  # descent order

        # 5. take topn score proposal
        topn_indices = indices[:pre_nms_topn]

        # 6. apply nms (e.g. threshold = 0.7)
        # proposals_boxes_c : [x, y, x`, y`, class]
        proposals_boxes_c = np.hstack((proposals_boxes[topn_indices], pos_score[topn_indices]))
        keep = non_max_suppression(proposals_boxes[topn_indices], pos_score[topn_indices], nms_thresh)

        # 7. take after_nms_topn (e.g. 300)
        if post_nms_topn > 0:
            keep = keep[:post_nms_topn]

        # 8. return the top proposals (-> RoIs top)
        proposals_boxes = proposals_boxes_c[keep, :-1]
        scores = proposals_boxes_c[keep, -1]

        return proposals_boxes, scores


class ROIPooling(nn.Module):
    def __init__(self, size=(7, 7), spatial_scale=1.0 / 16):
        super(ROIPooling, self).__init__()
        self.adaptivepool = nn.AdaptiveMaxPool2d(size)
        self.spatial_scale = spatial_scale

    def forward(self, img_features, roi_boxes):
        roi_boxes *= self.spatial_scale  # scale anchors to feature map size
        roi_boxes = roi_boxes.astype('int')
        output = []
        for bbox in roi_boxes:
            img_crop = img_features[:, :, bbox[1]:(bbox[3] + 1), bbox[0]:(bbox[2] + 1)]  # is this right?
            img_warped = self.adaptivepool(img_crop)
            output.append(img_warped)
        return torch.cat(output, 0)


class FRCNN(nn.Module):
    def __init__(self, batch_size=128, fg_fraction=0.25,
                 fg_threshold=0.5, bg_threshold=None,
                 num_classes=21):
        super(FRCNN, self).__init__()
        self.roi_pool = ROIPooling()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out_class = nn.Linear(4096, 21)
        self.out_regr = nn.Linear(4096, 4 * 21)

    def forward(self, img_features, roi_boxes):
        roi_pools = self.roi_pool(img_features, roi_boxes)
        x = roi_pools.view(roi_pools.size(0), -1)
        x = F.relu(self.fc1(x), True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), True)
        x = self.dropout(x)
        out_class = self.out_class(x)
        out_regr = self.out_regr(x)
        return out_class, out_regr

    def get_frcnn_targets(self, roi_boxes, targets, test):
        """
        Arguments:
            prop_boxes (Tensor) : (# proposal boxes , 4)
            gt_boxes_c (Ndarray) : (# gt boxes, 5) [x, y, x`, y`, class]
            test (Bool) : True or False
            args (argparse.Namespace) : global arguments

        Return:
            labels (Ndarray) : (256,)
            roi_boxes_c[:, :-1] : (256, 4)
            targets (Ndarray) : (256, 21 * 4)
            bbox_inside_weights (Ndarray) : (256, 21 * 4)
        """
        include_gt = True
        fg_fraction = 0.25
        fg_threshold = 0.5
        bg_threshold = (0.1, 0.5)
        frcnn_batch_size = 256
        target_bbs = boxes_to_numpy(targets)
        target_labels = classes_to_numpy(targets)

        all_boxes = np.vstack((roi_boxes[0], target_bbs)) if include_gt and not test else roi_boxes
        zeros = np.zeros((all_boxes.shape[0], 1), dtype=all_boxes.dtype)
        all_boxes_c = np.hstack((all_boxes, zeros))

        num_images = 1

        # number of roi_boxes_c each per image
        rois_per_image = int(frcnn_batch_size / num_images) if not test else int(roi_boxes.shape[0] / num_images)
        # number of foreground roi_boxes_c per image
        fg_rois_per_image = int(np.round(rois_per_image * fg_fraction))

        # sample_rois

        # compute each overlaps with each ground truth
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_boxes_c[:, :-1], dtype=np.float),
            np.ascontiguousarray(target_bbs, dtype=np.float))

        # overlaps (iou, index of class)
        overlaps = overlaps.cpu().numpy()

        # assign a predicted box to each ground truth box
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = target_labels[gt_assignment]

        fg_indices = np.where(max_overlaps >= fg_threshold)[0]
        fg_rois_per_this_image = min(fg_rois_per_image, len(fg_indices))

        if len(fg_indices) > 0:
            fg_indices = np.random.choice(fg_indices, size=fg_rois_per_this_image)

        if not test:
            bg_indices = np.where((max_overlaps < bg_threshold[1]) &
                                  (max_overlaps >= bg_threshold[0]))[0]
        else:
            bg_indices = np.where((max_overlaps < bg_threshold[1]) &
                                  (max_overlaps >= 0))[0]

        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, len(bg_indices))

        if len(bg_indices) > 0:
            bg_indices = np.random.choice(bg_indices, size=bg_rois_per_this_image)

        keep_inds = np.append(fg_indices, bg_indices)

        labels = labels[keep_inds]
        roi_boxes_c = all_boxes_c[keep_inds]

        # background에 해당하는 label을 0으로 만들어준다
        labels[fg_rois_per_this_image:] = 0

        delta_boxes = bbox_transform(roi_boxes_c, target_bbs[gt_assignment[keep_inds], :])

        bbox_targets = np.zeros((len(labels), 4 * 21), dtype=np.float32)

        # foreground object index
        indices = np.where(labels > 0)[0]

        for index in indices:
            cls = int(labels[index])
            start = 4 * cls
            end = start + 4
            bbox_targets[index, start:end] = delta_boxes[index, :]

        if len(labels) > 0:
            return torch.Tensor(labels).long().squeeze(), roi_boxes_c, torch.Tensor(bbox_targets)
        return labels, roi_boxes_c, bbox_targets