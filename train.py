from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from criterion import RPNLoss, FRCNNLoss
from datasets import PascalVOC
from models import Base_CNN, RegionProposalNetwork, Classifier
from torch.nn.utils import clip_grad_norm


path = '/home/austin/data/VOC/VOC2007/'
tr_mean, tr_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = transforms.Compose([transforms.Scale(600),
                            transforms.ToTensor(),
                            transforms.Normalize(tr_mean, tr_std)])
voc = PascalVOC(root=path + 'JPEGImages', annFile=path + "pascal_train2007.json", transform=trans)
loader = DataLoader(voc)
base_cnn = Base_CNN()
rpn = RegionProposalNetwork()
classifier = Classifier()

base_cnn.cuda()
rpn.cuda()
classifier.cuda()
optimizer = Adam([dict(params=rpn.parameters()), dict(params=classifier.parameters())], lr=1e-3)

criterion = RPNLoss()
criterion2 = FRCNNLoss()

# train rpn for one epoch
pbar = tqdm(loader, total=len(loader))
for img, target in pbar:
    img = Variable(img).cuda()
    img_features = base_cnn(img)
    rpn_cls_prob, rpn_bbox_pred = rpn(img_features)

    roi_boxes, scores = rpn.get_roi_boxes(rpn_bbox_pred, rpn_cls_prob, img, target, test=False)

    labels, bbox_targets = rpn.get_rpn_targets(target, img)
    rpn_loss = criterion(rpn_cls_prob, rpn_bbox_pred, Variable(labels).cuda(), Variable(bbox_targets).cuda())

    frcnn_labels, roi_boxes, frcnn_bbox_targets = classifier.get_frcnn_targets(roi_boxes, target, test=False)
    if roi_boxes.shape[0] == 0:
        continue
    cls_pred, bb_pred = classifier(img_features, roi_boxes)
    frcnn_loss = criterion2(cls_pred, bb_pred, Variable(frcnn_labels).cuda(), Variable(frcnn_bbox_targets).cuda())
    total_loss = rpn_loss + frcnn_loss
    pbar.set_description(desc='rpn_loss {} | frcnn_loss {}'.format(rpn_loss.data[0], frcnn_loss.data[0]))
    total_loss.backward()
    rpn_norm = clip_grad_norm(rpn.parameters(), 10.)
    frcnn_norm = clip_grad_norm(classifier.parameters(), 10.)
    optimizer.step()

