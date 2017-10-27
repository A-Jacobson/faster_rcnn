from utils import convert_wh_bbox, recover_wh_bbox, scale_targets
import numpy as np


def test_convert_wh_bbox():
    assert np.array_equal(convert_wh_bbox([155, 96, 196, 174]),
                          np.array([155,  96, 351, 270]))


def test_recover_wh_bbox():
    assert np.array_equal(recover_wh_bbox([155,  96, 351, 270]),
                          np.array([155, 96, 196, 174]))


def test_scale_targets():
    original_dims = 200, 400
    scaled_dims = 400, 800
    original_bbox = np.array([[50, 50, 100, 100]])
    scaled_bbox = np.array([[100, 100, 200, 200]])
    assert np.array_equal(scale_targets(original_bbox, original_dims, scaled_dims), scaled_bbox)


def test_iou():
    """
    iou is between 0 and 1
    :return:
    """
    pass

