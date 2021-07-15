import argparse
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
from torch.utils.data import DataLoader, BatchSampler
from torch import optim
from torch.autograd import Variable
from datetime import datetime
from segmentation_evalution import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torchinfo import summary


# https://www.bilibili.com/video/BV1yt4y1e7sZ?from=search&seid=407626673489639488


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


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 多机情况下：RANK表示多机情况下使用的第几台主机，单机情况：RANK表示使用的第几块GPU
        args.rank = int(os.environ["RANK"])
        # 多机情况下：WORLD_SIZE表示主机数，单机情况：WORLD_SIZE表示GPU数量
        args.world_size = int(os.environ['WORLD_SIZE'])
        # 多机情况下：LOCAL_RANK表示某台主机下的第几块设备，单机情况：与RANK相同
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    # 创建进程组
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # 等待每个GPU运行到此
    dist.barrier()


def main(args):
    init_distributed_mode(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 表示第几个GPU
    rank = args.rank

    # 在GPU_0上打印参数
    if rank == 0:
        print(args)

    # 获取分割类型数
    num_classes = cfg.DATASET[1]
    Cam_train = CamVidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    Cam_val = CamVidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

    # 为每个rank对应的进程分配训练样本
    Cam_train_sampler = DistributedSampler(Cam_train)
    Cam_val_sampler = DistributedSampler(Cam_val)
    # 将每个rank中的数据组成batch
    Cam_train_batch_sampler = BatchSampler(Cam_train_sampler, cfg.BATCH_SIZE, drop_last=True)
    Cam_val_batch_sampler = BatchSampler(Cam_val_sampler, cfg.BATCH_SIZE, drop_last=False)

    train_data = DataLoader(Cam_train, batch_sampler=Cam_train_batch_sampler, pin_memory=True, num_workers=4)
    val_data = DataLoader(Cam_val, batch_sampler=Cam_val_batch_sampler, pin_memory=True, num_workers=4)

    net = None
    if cfg.MODEL_TYPE == cfg.Model.FCN:
        net = FCN(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.SEG:
        net = Seg(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
        net = DeepLab(n_classes=num_classes).to(device)

    print('xxx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    main(opt)

    # if net is not None:
    #     # 创建并行模型
    #     net = DistributedDataParallel(net)
    #     criterion = nn.NLLLoss().to(device)
    #     optimizer = optim.Adam(net.parameters(), lr=cfg.BASE_LR)
    #
    #     # dump_file = open('net_dump.txt', 'w')
    #     # print(str(summary(net, input_size=(4, 3, 352, 480), device=device)), file=dump_file)
    #
    #     # train(net, train_data, val_data, criterion, optimizer, device)
    # else:
    #     print('Model is None')
