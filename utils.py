import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision import transforms

sns.set_style("white")


def get_output_dim(img, architecture='vgg16'):
    h, w = img.size()[-2:]
    if architecture is 'vgg16':
        stride = 2 ** 4
        return h // stride, w // stride  # integer divide by number of max pool lay


def show_img(tensor):
    return transforms.ToPILImage()(tensor)


def scale_bbs(img, target):
    """"
    :param img: C x H x W tensor, bbox form x, y, w, h
    :param target:
    :return:
    """
    scaled_target = copy.deepcopy(target)
    h_2 = img.size(1)
    w_2 = img.size(2)
    scale_x = w_2 / target[0]['width']
    scale_y = h_2 / target[0]['height']
    scaled_target[0]['scale'] = (scale_x, scale_y)
    for t in scaled_target:
        t['bbox'][0] = int(t['bbox'][0] * scale_x)
        t['bbox'][1] = int(t['bbox'][1] * scale_y)
        t['bbox'][2] = int(t['bbox'][2] * scale_x)
        t['bbox'][3] = int(t['bbox'][3] * scale_y)
    return scaled_target


def plot_with_bbs(img_tensor, target, classes=None):
    img = np.array(show_img(img_tensor))
    plt.imshow(img)
    colors = sns.color_palette(n_colors=21, desat=1)
    for t in target:
        color = colors[t['category_id']]
        rect = create_rect(t['bbox'], color=color)
        plt.gca().add_patch(rect)
        bbox_props = dict(boxstyle="square",
                          fc=colors[t['category_id']],
                          alpha=0.7)
        plt.gca().annotate(classes[t['category_id']],
                           xy=(t['bbox'][0] + 2,
                               t['bbox'][1] + 9),
                           bbox=bbox_props)
    plt.show()


def plot_with_bbs2(img_tensor, bboxes, labels=None, class_map=None):
    img = np.array(show_img(img_tensor))
    plt.imshow(img)
    colors = sns.color_palette(n_colors=20, desat=1)
    for i, box in enumerate(bboxes):
        color = colors[labels[i]] if isinstance(labels, np.ndarray) else 'blue'
        rect = create_rect(box, color=color)
        plt.gca().add_patch(rect)
        if not isinstance(labels, np.ndarray):
            continue
        name = class_map[labels[i]] if class_map else labels[i]
        bbox_props = dict(boxstyle="square",
                          fc=colors[labels[i]],
                          alpha=0.7)
        plt.gca().annotate(name,
                           xy=(box[0] + 2,
                               box[1] + 9),
                           bbox=bbox_props)
    plt.show()


def create_rect(bb, color='Blue', label=None):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=4, color=color, label=label)


def boxes_to_numpy(targets):
    target_bbs = np.zeros((len(targets), 4))
    for idx, label in enumerate(targets):
        # get the GT box coordinates, and reformat from (x, y, w , h) to x1, [x, y, x`, y`,]
        target_bbs[idx, 0] = label['bbox'][0][0]
        target_bbs[idx, 1] = label['bbox'][1][0]
        target_bbs[idx, 2] = label['bbox'][2][0]
        target_bbs[idx, 3] = label['bbox'][3][0]
    return target_bbs


def classes_to_numpy(targets):
    classes = np.zeros((len(targets), 1))
    for idx, label in enumerate(targets):
        classes[idx] = label['category_id'][0]
    return classes
