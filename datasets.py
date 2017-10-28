import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from utils import scale_targets, convert_wh_bbox


class PascalVOC(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, transform=None):
        self.root = os.path.join(root, 'JPEGImages')
        self.coco = COCO(os.path.join(root, "pascal_train2007.json"))
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.classes = {0: 'background',
                        1: 'aeroplane',
                        2: 'bicycle',
                        3: 'bird',
                        4: 'boat',
                        5: 'bottle',
                        6: 'bus',
                        7: 'car',
                        8: 'cat',
                        9: 'chair',
                        10: 'cow',
                        11: 'diningtable',
                        12: 'dog',
                        13: 'horse',
                        14: 'motorbike',
                        15: 'person',
                        16: 'pottedtable',
                        17: 'sheep',
                        18: 'sofa',
                        19: 'train',
                        20: 'tvmonitor'}

    def targets_to_numpy(self, targets):
        """
        convert mscoco targets dictionary to numpy array (N, x1, y1, x2, y1, class)
        """
        target_array = np.zeros((len(targets), 5))
        for i, target in enumerate(targets):
            bbox = convert_wh_bbox(target['bbox'])
            target_array[i, 0:4] = bbox
            target_array[i, 4] = target['category_id']
        return target_array

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is [N, x1, y1, x2, y2, C]
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        target = self.targets_to_numpy(target)

        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        original_dims = img.size
        if self.transform is not None:
            img = self.transform(img)
            scaled_dims = img.size()[2], img.size()[1]  # 3, H, W
            target = scale_targets(target, original_dims, scaled_dims)
        return img, target

    def __len__(self):
        return len(self.ids)
