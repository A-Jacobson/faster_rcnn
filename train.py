from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from criterion import RPNLoss, FRCNNLoss
from datasets import PascalVOC
from models import BaseCNN, RegionProposalNetwork, Classifier, FasterRCNN
from torch.nn.utils import clip_grad_norm
from utils import AverageMeter
from config import *

transform = transforms.Compose([transforms.Scale(SCALE),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])
voc_dataset = PascalVOC(DATA_PATH, transform)
voc_loader = DataLoader(voc_dataset, shuffle=SHUFFLE)
base_cnn = BaseCNN(REQUIRES_GRAD)
rpn = RegionProposalNetwork(batch_size=RPN_BATCH_SIZE)
classifier = Classifier(batch_size=CLF_BATCH_SIZE)

frcnn = FasterRCNN(base_cnn, rpn, classifier)
frcnn.cuda()
optimizer = SGD(filter(lambda p: p.requires_grad, frcnn.parameters()), lr=LEARNING_RATE)

criterion1 = RPNLoss()
criterion2 = FRCNNLoss()

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(voc_loader, total=len(voc_loader))
    for image, target in pbar:
        print(image.size())
        target = target.squeeze(0).numpy()
        loss_meter = AverageMeter()
        rpn_meter = AverageMeter()
        frcnn_meter = AverageMeter()
        rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = frcnn(image)
        rpn_labels, rpn_bbox_targets, rpn_batch_indices = frcnn.get_rpn_targets(target)
        classifier_labels, delta_boxes, clf_batch_indices = frcnn.get_classifier_targets(target)

        rpn_loss = criterion1(rpn_cls_probs, rpn_bbox_deltas, Variable(rpn_labels).cuda(),
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

