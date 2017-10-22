import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import numpy as np
import seaborn as sns
sns.set_style("white")



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
    for t in scaled_target:
        t['bbox'][0] = int(t['bbox'][0] * scale_x)
        t['bbox'][1] = int(t['bbox'][1] * scale_y)
        t['bbox'][2] = int(t['bbox'][2] * scale_x)
        t['bbox'][3] = int(t['bbox'][3] * scale_y)
    return scaled_target


def plot_with_bbs(img_tensor, target, classes=None):
    img = np.array(show_img(img_tensor))
    plt.imshow(img)
    colors = sns.color_palette(n_colors=20, desat=1)
    for t in target:
        color = colors[t['category_id']]
        rect = create_rect(t['bbox'], color=color)
        plt.gca().add_patch(rect)
        bbox_props = dict(boxstyle="square",
                          fc=colors[t['category_id']],
                          alpha=0.7)
        plt.gca().annotate(classes[t['category_id']],
                           xy=(t['bbox'][0]+2,
                               t['bbox'][1]+9),
                           bbox=bbox_props)
    plt.show()


def create_rect(bb, color='Blue', label=None):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=4, color=color, label=label)



