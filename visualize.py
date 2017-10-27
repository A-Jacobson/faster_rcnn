import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import seaborn as sns
from utils import recover_wh_bbox
import torch

sns.set_style("white")


def show_img(tensor):
    return transforms.ToPILImage()(tensor)


def create_rect(bb, color='Blue', label=None):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, lw=2, color=color, label=label)


def plot_with_targets(img, targets, class_map=None):
    if torch.is_tensor(img):
        img = np.array(show_img(img))
    plt.imshow(img)
    colors = sns.color_palette(n_colors=20, desat=1)
    for target in targets:
        if len(target) == 5:
            label = int(target[4])
            color = colors[label]
            name = class_map[label] if class_map else label
            bbox_props = dict(boxstyle="square", fc=color, alpha=0.7)
            plt.gca().annotate(name, xy=(target[0] + 2, target[1] + 9), bbox=bbox_props)
        else:
            color = colors[0]
        box = create_rect(recover_wh_bbox(target[:4]), color=color)
        plt.gca().add_patch(box)
    plt.show()
