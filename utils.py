import numpy as np


def convert_wh_bbox(bbox):
    """
    converts an (x, y, width, height) bbox to an (x1, y1, x2, y2) bbox
    """
    new_bbox = np.zeros_like(bbox)
    new_bbox[0] = bbox[0]
    new_bbox[1] = bbox[1]
    new_bbox[2] = bbox[0] + bbox[2]
    new_bbox[3] = bbox[1] + bbox[3]
    return new_bbox


def recover_wh_bbox(bbox):
    """
    converts an (x1, y1, x2, y2) bbox to an (x, y, width, height)
    """
    wh_bbox = np.zeros_like(bbox)
    wh_bbox[0] = bbox[0]
    wh_bbox[1] = bbox[1]
    wh_bbox[2] = bbox[2] - bbox[0]
    wh_bbox[3] = bbox[3] - bbox[1]
    return wh_bbox


def recover_wh_bboxes(bboxes):
    """
    converts a VOC bounding box (x, y, width, height) to rcnn format (x1, y1, x2, y2)
    """
    wh_bboxes = np.copy(bboxes)
    wh_bboxes[:, 0] = wh_bboxes[:, 0]
    wh_bboxes[:, 1] = wh_bboxes[:, 1]
    wh_bboxes[:, 2] = wh_bboxes[:, 2] - wh_bboxes[:, 0]
    wh_bboxes[:, 3] = wh_bboxes[:, 3] - wh_bboxes[:, 1]
    return wh_bboxes


def scale_targets(targets, original_dims, scaled_dims):
    """
    fits target bboxes to scaled image.

    :param bbox: (N, x1, y1, x2, y2, C)
    :param original_dims: (H, W)
    :param scaled_dims: (H, W)
    :return: bboxes scaled to fit new image
    """
    original_height, original_width = original_dims
    scaled_height, scaled_width = scaled_dims
    scale_x = scaled_width / original_width
    scale_y = scaled_height / original_height
    scaled_targets = np.copy(targets)
    scaled_targets[:, 0] = targets[:, 0] * scale_x
    scaled_targets[:, 1] = targets[:, 1] * scale_y
    scaled_targets[:, 2] = targets[:, 2] * scale_x
    scaled_targets[:, 3] = targets[:, 3] * scale_y
    return scaled_targets


def iou(anchors, target):
    """
    computes the iou between a vector of boxes (N, 4) and a single target box (1, 4)
    """
    # intersection
    x1 = np.maximum(anchors[:, 0], target[0])
    y1 = np.maximum(anchors[:, 1], target[1])
    x2 = np.minimum(anchors[:, 2], target[2])
    y2 = np.minimum(anchors[:, 3], target[3])

    # itersection area
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # union area
    anchors_area = (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    target_area = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)
    union = (anchors_area + target_area - intersection)

    return intersection / union


def pairwise_iou(anchor_boxes, targets):
    """
    computes ious between an array of anchor boxes and an arrach of targets
    returns (len(anchor_boxes), len(targets)) matrix
    """
    pairwise_ious = np.zeros((len(anchor_boxes), len(targets)))
    for i, target in enumerate(targets):
        pairwise_ious[:, i] = iou(anchor_boxes, target)
    return pairwise_ious


def get_overlaps(anchor_boxes, targets):
    return pairwise_iou(anchor_boxes, targets)


def pairwise_iou_vect(boxes_a, boxes_b):
    """
    computes ious between an two arrays of boxes
    returns (len(boxes_a), len(boxes_b)) matrix
    """
    # intersection
    x1 = np.maximum(boxes_a[:, 0], boxes_b[:, 0].reshape(-1, 1))
    y1 = np.maximum(boxes_a[:, 1], boxes_b[:, 1].reshape(-1, 1))
    x2 = np.minimum(boxes_a[:, 2], boxes_b[:, 2].reshape(-1, 1))
    y2 = np.minimum(boxes_a[:, 3], boxes_b[:, 3].reshape(-1, 1))

    # itersection area
    # TODO: OUTPUTS ZEROS
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # union area
    boxes_a_area = (boxes_a[:, 2] - boxes_a[:, 0] + 1) * (boxes_a[:, 3] - boxes_a[:, 1] + 1)
    boxes_b_area = (boxes_b[:, 2] - boxes_b[:, 0] + 1) * (boxes_b[:, 3] - boxes_b[:, 1] + 1)
    union = (boxes_a_area + boxes_b_area.reshape(-1, 1) - intersection)
    return intersection / union


def non_max_suppression(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break
    boxes = boxes[pick]
    probs = probs[pick]
    return boxes, probs


def bbox_transform(anchor_boxes, target_boxes):
    """
    Computes bbox delta needed to change anchor box to target boxes.
    These deltas will be the targets for RPN to predict.

    t_x = (x-x_anchor)/w_a x_center
    t_y = (y-y_anchor)/w_a
    t_w = log(w/w_anchor)
    t_h = log(h/h_anchor)

    gt_x = (gtx-x_a) / wa
    gt_y = (gty-y_anchor)/w_a
    gt_w = log(gwt/w_anchor)
    gt_h = log(gth/h_anchor)

    :param anchor_boxes: in x1, y1, x2, y2 format
    :param target_boxes: in x1, y1, x2, y2 format
    :return:
    """
    anchor_boxes = anchor_boxes.astype(np.float32)
    target_boxes = target_boxes.astype(np.float32)

    anchor_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0] + 1.0  # x1 - x2 + 1
    anchor_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.0  # y1 - y2 + 1
    anchor_center_xs = anchor_boxes[:, 0] + 0.5 * anchor_widths
    anchor_center_ys = anchor_boxes[:, 1] + 0.5 * anchor_heights

    target_widths = target_boxes[:, 2] - target_boxes[:, 0] + 1.0  # x1 - x2 + 1
    target_heights = target_boxes[:, 3] - target_boxes[:, 1] + 1.0  # y1 - y2 + 1
    target_center_xs = target_boxes[:, 0] + 0.5 * target_widths
    target_center_ys = target_boxes[:, 1] + 0.5 * target_heights

    targets_dx = (target_center_xs - anchor_center_xs) / anchor_widths
    targets_dy = (target_center_ys - anchor_center_ys) / anchor_heights
    targets_dw = np.log(target_widths / anchor_widths)
    targets_dh = np.log(target_heights / anchor_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(anchor_boxes, deltas):
    """
    Adjusts Generated Anchor boxes by the deltas predicted by RPN.
    :param anchor_boxes: anchor boxes
    :param deltas: predicted deltas (output from rpn)
    :return: Boxes Predicted by RPN
    """

    widths = anchor_boxes[:, 2] - anchor_boxes[:, 0] + 1.0
    heights = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1.0
    center_x = anchor_boxes[:, 0] + 0.5 * widths
    center_y = anchor_boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_center_x = dx * widths[:, np.newaxis] + center_x[:, np.newaxis]
    pred_center_y = dy * heights[:, np.newaxis] + center_y[:, np.newaxis]
    pred_widths = np.exp(dw) * widths[:, np.newaxis]
    pred_height = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_center_x - 0.5 * pred_widths
    # y1
    pred_boxes[:, 1::4] = pred_center_y - 0.5 * pred_height
    # x2
    pred_boxes[:, 2::4] = pred_center_x + 0.5 * pred_widths
    # y2
    pred_boxes[:, 3::4] = pred_center_y + 0.5 * pred_height
    return pred_boxes


def clip_boxes(boxes, image_dims):
    """
    Clip boxes to image boundaries.
    image_dims (height, width)
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], image_dims[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], image_dims[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], image_dims[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], image_dims[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return boxes[keep]
