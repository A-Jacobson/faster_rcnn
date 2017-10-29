import os

# BASE_CNN_CONFIG
ARCHITECTURE = 'resnet'
REQUIRES_GRAD = True

# RPN CONFIG
RPN_BATCH_SIZE = 128
FEATURE_STRIDE = 16
ANCHOR_SCALES = (128, 156, 512)
POSITIVE_OVERLAP = 0.7
NEGATIVE_OVERLAP = 0.3
RPN_FOREGROUND_FRACTION = 0.5
PRE_NMS_LIMIT = 6000
POST_NMS_LIMIT = 2000  # 300 at test time
NMS_THRESHOLD = 0.7
REG_LOSS_WEIGHT = 10.

# CLASSIFIER CONFIG
CLF_BATCH_SIZE = 128
CLF_FOREGROUND_FRACTION = 0.25
CLF_FOREGROUND_THRESHOLD = 0.5
CLF_BACKGROUND_THRESHOLD = 0.5
NUM_CLASSES = 21

# TRAINING CONFIG
DATA_PATH = '/home/austin/data/VOC/VOC2007/'
RESUME_PATH = None
RESUME_EPOCH = int(RESUME_PATH.split('/')[-1][0]) if RESUME_PATH else 0
LIMIT = None
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SCALE = 300
NUM_EPOCHS = 8
LEARNING_RATE = 1e-5  # 240 epochs 240k iters
# MOMENTUM = 0.9
SHUFFLE = True
CUDA = True  # would we ever tain this without cuda?
CLF_LOSS_WEIGHT = 10.
EXP_NAME = 'EXP3'
WEIGHT_DIR = os.path.join('weights', EXP_NAME)

