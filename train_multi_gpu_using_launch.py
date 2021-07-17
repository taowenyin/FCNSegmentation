import argparse
import torch
import torch.distributed as dist
import Utils.multi_gpu as multi_gpu
import Utils.train_eval as train_eval

from Models.FCN import FCN
from Models.Seg import Seg
from Models.DeepLab import DeepLab
from torchvision.models.segmentation import DeepLabV3, deeplabv3_resnet101
from dataset import CamVidDataset
from torch import nn
from torch.utils.data import DataLoader, BatchSampler
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from datetime import datetime
from segmentation_evalution import *
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary
from torch.nn import SyncBatchNorm


# https://www.bilibili.com/video/BV1yt4y1e7sZ?from=search&seid=407626673489639488
# python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_gpu_using_launch.py


def main(args):
    multi_gpu.init_distributed_mode(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 表示第几个GPU
    rank = args.rank

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
        # net = deeplabv3_resnet101(pretrained=False, num_classes=num_classes).to(device)
        net = DeepLab(n_classes=num_classes).to(device)

    if net is None:
        print('Model is None')
        raise NotImplementedError

    # 没有加载预训练权重，那么就要在不同的GPU中加载相同的权重
    # 在GPU0中保存默认权重
    if rank == 0:
        if cfg.MODEL_TYPE == cfg.Model.FCN:
            torch.save(net.state_dict(), './Results/weights/FCN_best.pth')
        elif cfg.MODEL_TYPE == cfg.Model.SEG:
            torch.save(net.state_dict(), './Results/weights/SEG_best.pth')
        elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
            torch.save(net.state_dict(), './Results/weights/DeepLab_best.pth')

    # 等待每个GPU运行到此
    dist.barrier()

    # 每个GPU中的模型载入相同的权重
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    if cfg.MODEL_TYPE == cfg.Model.FCN:
        net.load_state_dict(torch.load("./Results/weights/FCN_best.pth", map_location=device))
    elif cfg.MODEL_TYPE == cfg.Model.SEG:
        net.load_state_dict(torch.load("./Results/weights/SEG_best.pth", map_location=device))
    elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
        net.load_state_dict(torch.load("./Results/weights/DeepLab_best.pth", map_location=device))

    # 等待每个GPU运行到此
    dist.barrier()

    if args.syncBN:
        # 把BN改成同步结构
        net = SyncBatchNorm.convert_sync_batchnorm(net).to(device)

    # 创建DDP模型
    net = DistributedDataParallel(net, device_ids=[args.gpu])

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.BASE_LR)
    # Poly学习率更新策略
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cfg.BASE_LR * (1 - epoch / cfg.EPOCH_NUMBER)**cfg.LR_POWER
    )
    # # 每训练50次学习率下降一半
    # if epoch % 50 == 0 and epoch != 0:
    #     for group in optimizer.param_groups:
    #         group['lr'] *= 0.5

    # 分割评价对象
    evalution = SegmentationEvalution(cfg.DATASET[1])

    # dump_file = open('net_dump.txt', 'w')
    # print(str(summary(net, input_size=(4, 3, 352, 480), device=device)), file=dump_file)

    # 保存最好权重
    best = [0]
    for epoch in range(cfg.EPOCH_NUMBER):
        print('Rank {} Epoch is [{}/{}]'.format(args.rank, epoch + 1, cfg.EPOCH_NUMBER))
        # 每次迭代都重新打乱所有数据
        Cam_train_sampler.set_epoch(epoch)

        # 训练
        train_eval.train_one_epoch(net, train_data, evalution, criterion, optimizer, device)

        # 更新学习率
        scheduler.step()

        # 在GPU0上进行指标计算
        if rank == 0:
            # 计算训练指标
            metrics = evalution.get_scores()
            for k, v in metrics[0].items():
                print(k, v)
            # 保存最优权重
            train_miou = metrics[0]['mIou: ']
            if max(best) <= train_miou:
                best.append(train_miou)
                if cfg.MODEL_TYPE == cfg.Model.FCN:
                    torch.save(net.state_dict(), './Results/weights/FCN_best.pth')
                elif cfg.MODEL_TYPE == cfg.Model.SEG:
                    torch.save(net.state_dict(), './Results/weights/SEG_best.pth')
                elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
                    torch.save(net.state_dict(), './Results/weights/DeepLab_best.pth')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~')

        # 验证
        train_eval.evaluate_one_epoch(net, val_data, evalution, criterion, device)

        # 在GPU0上进行指标计算
        if rank == 0:
            # 计算训练指标
            metrics = evalution.get_scores()
            for k, v in metrics[0].items():
                print(k, v)
            print('==========================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    opt = parser.parse_args()
    main(opt)
