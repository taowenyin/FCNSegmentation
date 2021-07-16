from enum import Enum

TRAIN_ROOT = './Datasets/CamVid/train'
TRAIN_LABEL = './Datasets/CamVid/train_labels'
VAL_ROOT = './Datasets/CamVid/val'
VAL_LABEL = './Datasets/CamVid/val_labels'
TEST_ROOT = './Datasets/CamVid/test'
TEST_LABEL = './Datasets/CamVid/test_labels'

class_dict_path = './Datasets/CamVid/class_dict.csv'


# 模型类型
class Model(Enum):
    FCN = 1
    SEG = 2
    DEEP_LAB = 3


BATCH_SIZE = 4

DATASET = ['CamVid', 12]
# DATASET = ['PASCAL VOC', 21]

# 模型类型
MODEL_TYPE = Model.FCN

EPOCH_NUMBER = 200
# EPOCH_NUMBER = 100

# 基准学习率
BASE_LR = 7e-3

# poly学习率策略
LR_POWER = 0.9

# 由于Torch中裁剪函数要求(height, width)，因此为352x480，
# 因为后续在进行下采样后第5次采样为原始图像的1/32，因此原先的360不能整除，而352可以
crop_size = (352, 480)
