import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metrics import iou


class RegionProposalNetwork(nn.Module):
    def __init__(self, num_anchors):
        super(RegionProposalNetwork, self).__init__()
        self.rpn_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_classify = nn.Conv2d(512, num_anchors, kernel_size=1)
        self.conv_bbox_regr = nn.Conv2d(512, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        conv1 = F.relu(self.rpn_conv1(x), True)
        classify = F.sigmoid(self.conv_classify(conv1))
        roi_boxes = self.conv_bbox_regr(conv1)
        return [classify, roi_boxes, x]


def get_output_dim(img, architecture='vgg16'):
    h, w = img.size()[-2:]
    if 'vgg16':
        return h//2**5, w//2**5  # integer divide by number of max pool layers


def calculate_rpn_targets(img, target, architecture='vgg16'):
    """
    :param img:
    :param target:
    :param architecture:
    :return:
    """
    # set defaults from paper

    downscale = 16
    anchor_sizes = [128, 256, 512]
    anchor_ratios = [[1, 1],
                     [1. / math.sqrt(2), 2. / math.sqrt(2)],
                     [2. / math.sqrt(2), 1. / math.sqrt(2)]]
    rpn_max_overlap = 0.7
    rpn_min_overlap = 0.3
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # calculate the output map size based on the network architecture
    output_height, output_width = get_output_dim(img, architecture)

    n_anchoratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(target)

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, label in enumerate(target):
        # get the GT box coordinates, and reformat from (x, y, h , w) to (x1, x2, y1, y2)
        gta[bbox_num, 0] = label['bbox'][0]
        gta[bbox_num, 1] = label['bbox'][2]
        gta[bbox_num, 2] = label['bbox'][1]
        gta[bbox_num, 3] = label['bbox'][3]

    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchoratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > img.size()[0]:
                    continue

                for jy in range(output_height):

                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > img.size()[1]:
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):

                        # get IOU of the current GT box and the current anchor box
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if target[bbox_num]['category_id'] != 0:  # 0 = background class

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > rpn_max_overlap:
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if rpn_min_overlap < curr_iou < rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchoratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchoratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchoratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchoratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchoratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchoratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchoratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # we ensure that every bbox has at least one positive RPN region

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue

            y_is_box_valid[best_anchor_for_bbox[idx, 0],
                           best_anchor_for_bbox[idx, 1],
                           best_anchor_for_bbox[idx, 2] + n_anchoratios * best_anchor_for_bbox[idx, 3]] = 1

            y_rpn_overlap[best_anchor_for_bbox[idx, 0],
                          best_anchor_for_bbox[idx, 1],
                          best_anchor_for_bbox[idx, 2] + n_anchoratios * best_anchor_for_bbox[idx, 3]] = 1

            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchoratios * best_anchor_for_bbox[idx, 3])

            y_rpn_regr[best_anchor_for_bbox[idx, 0],
                       best_anchor_for_bbox[idx, 1],
                       start:start + 4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)
