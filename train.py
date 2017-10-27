from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from criterion import RPNLoss, FRCNNLoss
from datasets import PascalVOC
from models import BaseCNN, RegionProposalNetwork, Classifier, FasterRCNN
from torch.nn.utils import clip_grad_norm


path = '/home/austin/data/VOC/VOC2007/'
tr_mean, tr_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
trans = transforms.Compose([transforms.Scale(300),
                            transforms.ToTensor(),
                            transforms.Normalize(tr_mean, tr_std)])
voc = PascalVOC(root=path, transform=trans)
base_cnn = BaseCNN()
rpn = RegionProposalNetwork()
classifier = Classifier()

frcnn = FasterRCNN(base_cnn, rpn, classifier)
frcnn.cuda()
optimizer = Adam([dict(params=rpn.parameters()), dict(params=classifier.parameters())], lr=1e-3)

criterion = RPNLoss()
criterion2 = FRCNNLoss()

# train rpn for one epoch
pbar = tqdm(voc, total=len(voc))
for img, target in pbar:
    img = Variable(img.unsqueeze(0)).cuda()
    rpn_cls_probs, rpn_bbox_deltas, pred_label, pred_bbox_deltas = frcnn(img)
    rpn_labels, rpn_bbox_targets, rpn_batch_indices = frcnn.get_rpn_targets(target)
    classifier_labels, delta_boxes, clf_batch_indices = frcnn.get_classifier_targets(target)

    rpn_batch_indices = rpn_batch_indices.cuda()
    clf_batch_indices = clf_batch_indices.cuda()
    rpn_loss = criterion(rpn_cls_probs, rpn_bbox_deltas, Variable(rpn_labels).cuda(),
                         Variable(rpn_bbox_targets).cuda(), rpn_batch_indices)

    frcnn_loss = criterion2(pred_label,pred_bbox_deltas, Variable(classifier_labels).cuda(),
                            Variable(delta_boxes).cuda(), clf_batch_indices)

    total_loss = rpn_loss + frcnn_loss
    pbar.set_description(desc='rpn_loss {} | frcnn_loss {}'.format(rpn_loss.data[0], frcnn_loss.data[0]))
    total_loss.backward()
    # rpn_norm = clip_grad_norm(rpn.parameters(), 10.)
    # frcnn_norm = clip_grad_norm(classifier.parameters(), 10.)
    optimizer.step()

