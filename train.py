import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from Models.FCN import FCN
from Models.Seg import Seg
from Models.DeepLab import DeepLab
from dataset import CamVidDataset
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from segmentation_evalution import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torchinfo import summary


# 设置学习率策略为Poly
def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, train_data, val_data, criterion, optimizer, device):
    # 保存最好权重
    best = [0]
    net = model.train()
    evalution = SegmentationEvalution(cfg.DATASET[1])

    for epoch in range(cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        # 重置评价结果
        evalution.reset()

        adjust_learning_rate_poly(optimizer, epoch, cfg.EPOCH_NUMBER, cfg.BASE_LR, 0.9)
        # # 每训练50次学习率下降一半
        # if epoch % 50 == 0 and epoch != 0:
        #     for group in optimizer.param_groups:
        #         group['lr'] *= 0.5

        train_loss = 0

        for i, sample in enumerate(train_data):
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))

            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 去每个像素的最大值索引
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            # 获取真实的标签
            true_label = img_label.data.cpu().numpy()

            # 指标计算
            evalution.update(true_label, pre_label)

        metrics = evalution.get_scores()
        for k, v in metrics[0].items():
            print(k, v)
        print('Train Loss: ', train_loss)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~')

        train_miou = metrics[0]['mIou: ']
        if max(best) <= train_miou:
            best.append(train_miou)
            if cfg.MODEL_TYPE == cfg.Model.FCN:
                torch.save(net.state_dict(), './Results/weights/FCN_best.pth')
            elif cfg.MODEL_TYPE == cfg.Model.SEG:
                torch.save(net.state_dict(), './Results/weights/SEG_best.pth')
            elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
                torch.save(net.state_dict(), './Results/weights/DeepLab_best.pth')

        evaluate(model, val_data, criterion, device)


def evaluate(model, eval_data, criterion, device):
    net = model.eval()
    evalution = SegmentationEvalution(cfg.DATASET[1])

    eval_loss = 0

    prec_time = datetime.now()
    for i, sample in enumerate(eval_data):
        img_data = Variable(sample['img'].to(device))
        img_label = Variable(sample['label'].to(device))

        # 训练
        out = net(img_data)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, img_label)
        eval_loss += loss.item()

        # 去每个像素的最大值索引
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        # 获取真实的标签
        true_label = img_label.data.cpu().numpy()

        # 指标计算
        evalution.update(true_label, pre_label)

    metrics = evalution.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    print('Evaluate Loss: ', eval_loss)

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)
    print('==========================')


if __name__ == '__main__':
    # GPU数量
    gpu_list = np.arange(torch.cuda.device_count())

    Cam_train = CamVidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    Cam_val = CamVidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = cfg.DATASET[1]

    train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)

    net = None
    if cfg.MODEL_TYPE == cfg.Model.FCN:
        net = FCN(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.SEG:
        net = Seg(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
        net = DeepLab(n_classes=num_classes).to(device)

    if net is not None:
        # 创建并行模型
        # net = DistributedDataParallel(net, device_ids=gpu_list)
        criterion = nn.NLLLoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr=cfg.BASE_LR)

        # dump_file = open('net_dump.txt', 'w')
        # print(str(summary(net, input_size=(4, 3, 352, 480), device=device)), file=dump_file)

        train(net, train_data, val_data, criterion, optimizer, device)
    else:
        print('Model is None')
