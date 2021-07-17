import torch
import Utils.train_eval as train_eval

from Models.FCN import FCN
from Models.Seg import Seg
from Models.DeepLab import DeepLab
from torchvision.models.segmentation import deeplabv3_resnet101
from dataset import CamVidDataset
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from segmentation_evalution import *
from torchinfo import summary


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 获取分割类型数
    num_classes = cfg.DATASET[1]
    Cam_train = CamVidDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    Cam_val = CamVidDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

    train_data = DataLoader(Cam_train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                            pin_memory=True, num_workers=4, drop_last=True)
    val_data = DataLoader(Cam_val, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=4, drop_last=False)

    net = None
    if cfg.MODEL_TYPE == cfg.Model.FCN:
        net = FCN(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.SEG:
        net = Seg(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
        net = DeepLab(n_classes=num_classes).to(device)
        # net = deeplabv3_resnet101(pretrained=False, num_classes=num_classes).to(device)

    if net is None:
        print('Model is None')
        raise NotImplementedError

    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.BASE_LR)
    # Poly学习率更新策略
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cfg.BASE_LR * (1 - epoch / cfg.EPOCH_NUMBER) ** cfg.LR_POWER
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
        # 训练
        train_eval.train_one_epoch(net, train_data, evalution, criterion, optimizer, device, epoch)

        # 更新学习率
        scheduler.step()

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
        metrics = evalution.get_scores()
        for k, v in metrics[0].items():
            print(k, v)
        print('==========================')
