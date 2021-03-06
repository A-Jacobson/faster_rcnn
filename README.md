# Faster RCNN pytorch
[WIP] Faster RCNN pytorch

- [x] Load VOC
- [x] bbox scaling
- [x] bbox plotting
- [x] VGG features
- [x] Anchor generation
- [x] RPN
- [x] Roi Pooling
- [x] Non maximum suppression
- [x] RPN loss
- [x] Backward pass
- [x] refactor frcnn into one object
- [x] remove label dependency to generate boxes
- [x] Resnet34 Features (memory savings?)
- [ ] clip proposals before detection loss
- [ ] Train 100 epochs at scale (300) - in-progress
- [ ] MAP functions
- [ ] Data Augmentation
- [ ] Train on VOC (in progress)
- [ ] Evaluate Results


## resources
- https://arxiv.org/abs/1506.01497
- https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/object_localization_and_detection.html
- https://github.com/yhenon/keras-frcnn/blob/master/train_frcnn.py
- https://github.com/mitmul/chainer-faster-rcnn/tree/v2/models
- https://github.com/chainer/chainercv/blob/master/chainercv/links/model/faster_rcnn/faster_rcnn.py