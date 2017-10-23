import numpy as np
import torch
from torch.autograd import Variable


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

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    # probs = probs[pick]
    return boxes


def unmap(data, count, inds, fill=0):
    """ Unmap boxes subset of item (data) back to the original set of items (of
    size count) """

    # make label data
    # map up to original set of anchors
    # labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    # bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def bbox_transform(pred_boxes, target_boxes):
    """
    Altered from original FasterRCNNpy code for use with COCO format
    Finds delta between pred_boxes and target boxes
    :param pred_boxes: in x1, y1, x2, y2 format
    :param target_boxes: in x1, y1, w, h format
    :return:
    """

    pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0] + 1.0  # x1 - x2 + 1
    pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1] + 1.0  # y1 - y2 + 1
    pred_center_xs = pred_boxes[:, 0] + 0.5 * pred_widths
    pred_center_ys = pred_boxes[:, 1] + 0.5 * pred_heights

    target_widths = target_boxes[:, 2] + 1.0
    target_heights = target_boxes[:, 3] + 1.0
    target_center_xs = target_boxes[:, 0] + 0.5 * target_widths
    target_center_ys = target_boxes[:, 1] + 0.5 * target_heights

    targets_dx = (target_center_xs - pred_center_xs) / pred_widths
    targets_dy = (target_center_ys - pred_center_ys) / pred_heights
    targets_dw = np.log(target_widths / pred_widths)
    targets_dh = np.log(target_heights / pred_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(anchor_boxes, deltas):
    """
    Adjusts Generated Anchor boxes by the deltas predicted by RPN.
    :param anchor_boxes: anchor boxes
    :param deltas: predicted deltas (output from rpn)
    :return: Boxes Predicted by RPN
    """
    if anchor_boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    anchor_boxes = anchor_boxes.astype(deltas.dtype, copy=False)

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


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


# torch tensors
def bbox_overlaps(a, bb):
    ## IOU
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(bb, np.ndarray):
        bb = torch.from_numpy(bb)

    oo = []

    for b in bb:
        x1 = a.select(1, 0).clone()
        x1[x1.lt(b[0])] = b[0]
        y1 = a.select(1, 1).clone()
        y1[y1.lt(b[1])] = b[1]
        x2 = a.select(1, 2).clone()
        x2[x2.gt(b[2])] = b[2]
        y2 = a.select(1, 3).clone()
        y2[y2.gt(b[3])] = b[3]

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        inter = torch.mul(w, h).float()
        aarea = torch.mul((a.select(1, 2) - a.select(1, 0) + 1), (a.select(1, 3) - a.select(1, 1) + 1)).float()
        barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        # intersection over union overlap
        o = torch.div(inter, (aarea + barea - inter))
        # set invalid entries to 0 overlap
        o[w.lt(0)] = 0
        o[h.lt(0)] = 0

        oo += [o]

    return torch.cat([o.view(-1, 1) for o in oo], 1)


def to_var(x):
    if isinstance(x, np.ndarray):
        return Variable(torch.from_numpy(x), requires_grad=False)
    elif torch.is_tensor(x):
        return Variable(x, requires_grad=True)
    elif isinstance(x, tuple):
        t = []
        for i in x:
            t.append(to_var(i))
        return t
    elif isinstance(x, Variable):
        return x
