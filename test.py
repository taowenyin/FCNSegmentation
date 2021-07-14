import torch
import torch.nn.functional as F
import time

from Models.FCN import FCN
from Models.Seg import Seg
from dataset import CamVidDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from segmentation_evalution import *


def test(model, test_data, device):
    net = model.eval()
    evalution = SegmentationEvalution(cfg.DATASET[1])
    time_meter = AverageMeter()

    for i, sample in enumerate(test_data):
        time_start = time.time()
        img_data = Variable(sample['img'].to(device))
        img_label = Variable(sample['label'].to(device))

        # 训练
        out = net(img_data)
        out = F.log_softmax(out, dim=1)

        # 去每个像素的最大值索引
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        # 获取真实的标签
        true_label = img_label.data.cpu().numpy()

        # 指标计算
        evalution.update(true_label, pre_label)

        time_meter.update(time.time() - time_start, n=img_data.size(0))

    metrics = evalution.get_scores()
    for k, v in metrics[0].items():
        print(k, v)

    # 计算分割速度
    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)


if __name__ == '__main__':
    Cam_test = CamVidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = cfg.DATASET[1]

    test_data = DataLoader(Cam_test, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

    net = None
    if cfg.MODEL_TYPE == cfg.Model.FCN:
        net = FCN(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.SEG:
        net = Seg(num_classes).to(device)

    if net is not None:
        if cfg.MODEL_TYPE == cfg.Model.FCN:
            net.load_state_dict(torch.load("./Results/weights/FCN_best.pth"))
        elif cfg.MODEL_TYPE == cfg.Model.SEG:
            net.load_state_dict(torch.load("./Results/weights/SEG_best.pth"))
        elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
            net.load_state_dict(torch.load("./Results/weights/DeepLab_best.pth"))

        test(net, test_data, device)
    else:
        print('Model is None')