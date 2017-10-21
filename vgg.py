from torchvision.models import vgg16_bn
from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

vgg_full = vgg16_bn(pretrained=True)
vggfeats = nn.Sequential(vgg_full.features)


class RegionProposalNetwork(nn.Module):
    def __init__(self, num_anchors):
        super(RegionProposalNetwork, self).__init__()
        self.rpn_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_classify = nn.Conv2d(512, num_anchors, kernel_size=1)
        self.conv_bbox_regr = nn.Conv2d(512, num_anchors*4, kernel_size=1)

    def forward(self, x):
        conv1 = F.relu(self.rpn_conv1(x), True)
        classify = F.sigmoid(self.conv_classify(conv1))
        roi_boxes = self.conv_bbox_regr(conv1)
        return [classify, roi_boxes, x]


def roi_pooling(img, rois, size=(7, 7), spatial_scale=1.0):
    output = []
    rois = rois.data.float()
    rois[:, 1:] *= spatial_scale
    rois = rois.long()
    for roi in rois:
        img_crop = img[:, :, roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        img_warped = F.adaptive_max_pool2d(img_crop, size)
        output.append(img_warped)
    return torch.cat(output, 0)

# input = Variable(torch.rand(1,1,10,10), requires_grad=True)
