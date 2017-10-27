from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchvision import transforms
from tqdm import tqdm

from criterion import RPNLoss, FRCNNLoss
from datasets import PascalVOC
from models import BaseCNN, RegionProposalNetwork, Classifier, FasterRCNN
from torch.nn.utils import clip_grad_norm
from utils import AverageMeter

num_epochs = 5
path = '/home/austin/data/VOC/VOC2007/'
tr_mean, tr_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = transforms.Compose([transforms.Scale(180),
                            transforms.ToTensor(),
                            transforms.Normalize(tr_mean, tr_std)])
voc = PascalVOC(root=path, transform=trans)
base_cnn = BaseCNN()
rpn = RegionProposalNetwork(batch_size=128)
classifier = Classifier(batch_size=128)

frcnn = FasterRCNN(base_cnn, rpn, classifier)
frcnn.cuda()
optimizer = SGD(filter(lambda p: p.requires_grad, frcnn.parameters()), lr=1e-3)

criterion = RPNLoss()
criterion2 = FRCNNLoss()

# train rpn for one epoch
for epoch in range(num_epochs):
    pbar = tqdm(voc, total=len(voc))
    for img, target in pbar:
        loss_meter = AverageMeter()
        rpn_meter = AverageMeter()
        frcnn_meter = AverageMeter()
        img = Variable(img.unsqueeze(0)).cuda()
        rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = frcnn(img)
        rpn_labels, rpn_bbox_targets, rpn_batch_indices = frcnn.get_rpn_targets(target)
        classifier_labels, delta_boxes, clf_batch_indices = frcnn.get_classifier_targets(target)

        rpn_loss = criterion(rpn_cls_probs, rpn_bbox_deltas, Variable(rpn_labels).cuda(),
                             Variable(rpn_bbox_targets).cuda(), rpn_batch_indices.cuda())

        frcnn_loss = criterion2(pred_label, pred_bbox_deltas, Variable(classifier_labels).cuda(),
                                Variable(delta_boxes).cuda(), clf_batch_indices.cuda())
        total_loss = rpn_loss + frcnn_loss


        rpn_meter.update(rpn_loss.data[0])
        frcnn_meter.update(frcnn_loss.data[0])
        loss_meter.update(total_loss.data[0])

        pbar.set_description(desc='loss {:.4g} | rpn loss {:.4f} | frcnn loss {:.4f}'.format(loss_meter.avg,
                                                                                            rpn_meter.avg,
                                                                                            frcnn_meter.avg))
        total_loss.backward()
        optimizer.step()

