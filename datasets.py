import os

from torch.utils.data import Dataset
from PIL import Image
from utils import scale_bbs


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

    def __init__(self, root, annFile, transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.classes = {1: 'aeroplane',
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
                        20: 'tvmonitor',
                        21: 'ignore'}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        w = img.size[0]
        h = img.size[1]
        target[0]['w'] = w
        target[0]['h'] = h
        if self.transform is not None:
            img = self.transform(img)
            target = scale_bbs(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)
