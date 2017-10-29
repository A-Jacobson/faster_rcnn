import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import gc


from config import *
from criterion import RPNLoss, FRCNNLoss
from datasets import PascalVOC
from models import BaseCNN, RegionProposalNetwork, Detector, FasterRCNN
from utils import AverageMeter, save_checkpoint
from visualize import plot_with_targets, recover_imagenet

TEST_IMG_IDX = 3

if not os.path.exists(WEIGHT_DIR):
    os.mkdir(WEIGHT_DIR)

cudnn.benchmark = True

transform = transforms.Compose([transforms.Scale(SCALE),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])

voc_dataset = PascalVOC(DATA_PATH, transform, LIMIT)
voc_loader = DataLoader(voc_dataset, shuffle=SHUFFLE, pin_memory=True)
basecnn = BaseCNN(ARCHITECTURE, REQUIRES_GRAD)
rpn = RegionProposalNetwork(batch_size=RPN_BATCH_SIZE)
detector = Detector(batch_size=CLF_BATCH_SIZE)

frcnn = FasterRCNN(basecnn, rpn, detector)
optimizer = Adam(filter(lambda p: p.requires_grad, frcnn.parameters()), lr=LEARNING_RATE)

if RESUME_PATH:
    experiment = torch.load(RESUME_PATH)
    frcnn.load_state_dict(experiment['model_state'])
    optimizer.load_state_dict(experiment['optimizer_state'])
    frcnn.basecnn.finetune()

frcnn.cuda()

criterion1 = RPNLoss(REG_LOSS_WEIGHT)
criterion2 = FRCNNLoss(REG_LOSS_WEIGHT)


def train(model, num_epochs, resume_epoch):
    loss_meter = AverageMeter()
    rpn_meter = AverageMeter()
    frcnn_meter = AverageMeter()
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        pbar = tqdm(voc_loader, total=len(voc_loader), leave=True)
        for image, target in pbar:
            gc.collect()
            image = Variable(image).cuda(async=True)
            target = target.squeeze(0).numpy()
            rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = frcnn(image)
            proposal_boxes, _ = frcnn.get_rpn_proposals()

            if len(proposal_boxes) == 0:
                continue

            rpn_labels, rpn_bbox_targets, rpn_batch_indices = frcnn.get_rpn_targets(target)
            detector_labels, delta_boxes, clf_batch_indices = frcnn.get_detector_targets(target)
            rpn_loss = criterion1(rpn_cls_probs, rpn_bbox_deltas,
                                  Variable(rpn_labels, requires_grad=False).cuda(),
                                  Variable(rpn_bbox_targets, requires_grad=False).cuda(),
                                  rpn_batch_indices.cuda())
            frcnn_loss = criterion2(pred_label, pred_bbox_deltas,
                                    Variable(detector_labels, requires_grad=False).cuda(),
                                    Variable(delta_boxes, requires_grad=False).cuda(),
                                    clf_batch_indices.cuda())
            total_loss = rpn_loss + frcnn_loss

            rpn_meter.update(rpn_loss.data[0])
            frcnn_meter.update(frcnn_loss.data[0])
            loss_meter.update(total_loss.data[0])

            pbar.set_description(desc='loss {:.4f} | rpn loss {:.4f} | frcnn loss {:.4f}'.format(loss_meter.avg,
                                                                                                 rpn_meter.avg,
                                                                                                 frcnn_meter.avg))
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            save_checkpoint(frcnn.state_dict(),
                            optimizer.state_dict(),
                            os.path.join(WEIGHT_DIR, "{}_{:.1e}_{:.4f}.pt".format(epoch + 1 + resume_epoch,
                                                                                  LEARNING_RATE, loss_meter.avg)))

        loss_meter.reset()
        rpn_meter.reset()
        frcnn_meter.reset()

        # checkpoint


train(frcnn, NUM_EPOCHS, RESUME_EPOCH)
test_img, test_target = voc_dataset[TEST_IMG_IDX]
plot_with_targets(recover_imagenet(test_img), test_target)
pred_targets = frcnn.get_predictions(Variable(test_img.unsqueeze(0)).cuda(), ignore_background=True)
plot_with_targets(recover_imagenet(test_img), pred_targets)
