import warnings

import torch
from torch.backends import cudnn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import *
from criterion import RPNLoss, FRCNNLoss
from datasets import PascalVOC
from models import BaseCNN, RegionProposalNetwork, Classifier, FasterRCNN
from utils import AverageMeter, save_checkpoint, bbox_transform_inv
from visualize import plot_with_targets

cudnn.benchmark = True

warnings.filterwarnings("error")

transform = transforms.Compose([transforms.Scale(SCALE),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])
voc_dataset = PascalVOC(DATA_PATH, transform)
voc_loader = DataLoader(voc_dataset, shuffle=SHUFFLE)
base_cnn = BaseCNN(ARCHITECTURE, REQUIRES_GRAD)
rpn = RegionProposalNetwork(batch_size=RPN_BATCH_SIZE)
classifier = Classifier(batch_size=CLF_BATCH_SIZE)

frcnn = FasterRCNN(base_cnn, rpn, classifier)
frcnn.cuda()
optimizer = Adam(filter(lambda p: p.requires_grad, frcnn.parameters()), lr=LEARNING_RATE)

if RESUME_PATH:
    experiment = torch.load(RESUME_PATH)
    frcnn.load_state_dict(experiment['model_state'])
    optimizer.load_state_dict(experiment['optimizer_state'])

criterion1 = RPNLoss(REG_LOSS_WEIGHT)
criterion2 = FRCNNLoss(REG_LOSS_WEIGHT)

for epoch in range(NUM_EPOCHS):
    pbar = tqdm(voc_loader, total=len(voc_loader))
    loss_meter = AverageMeter()
    rpn_meter = AverageMeter()
    frcnn_meter = AverageMeter()
    for image, target in pbar:
        image = Variable(image).cuda()
        target = target.squeeze(0).numpy()
        rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = frcnn(image)
        proposal_boxes, _ = frcnn.get_rpn_proposals()

        if len(proposal_boxes) == 0:
            continue

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

        pbar.set_description(desc='loss {:.4f} | rpn loss {:.4f} | frcnn loss {:.4f}'.format(loss_meter.avg,
                                                                                             rpn_meter.avg,
                                                                                             frcnn_meter.avg))
        total_loss.backward()
        optimizer.step()

    # checkpoint
    if not os.path.exists(WEIGHT_DIR):
        os.mkdir(WEIGHT_DIR)
    save_checkpoint(frcnn.state_dict(),
                    optimizer.state_dict(),
                    os.path.join(WEIGHT_DIR, "{}_{:.2f}_{:.4f}".format(epoch+1+RESUME_EPOCH,
                                                                       LEARNING_RATE, loss_meter.avg)))
    # idx = 1
    # img, targets = voc_dataset[idx]
    # rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = frcnn(Variable(img).unsqueeze(0))
    # proposal_boxes, _ = frcnn.get_rpn_proposals()
    # class_boxes = bbox_transform_inv(proposal_boxes, pred_bbox_deltas)
    #
    # plot_with_targets(img, targets)
    # plot_with_targets(img, proposal_boxes)
    # plot_with_targets(img, class_boxes)
