import torch
import torch.nn.functional as F
import Utils.multi_gpu as multi_gpu

from torch.autograd import Variable
from datetime import datetime


def train_one_epoch(model, train_data, evalution, criterion, optimizer, device):
    model.train()
    # 重置评价结果
    evalution.reset()
    # 保存训练损失
    train_loss = torch.zeros(1).to(device)
    # 清空梯度
    optimizer.zero_grad()

    for i, sample in enumerate(train_data):
        optimizer.step()
        optimizer.zero_grad()

        img_data = Variable(sample['img'].to(device))
        img_label = Variable(sample['label'].to(device))

        # 训练
        # out = model(img_data)['out']
        out = model(img_data)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, img_label)
        loss.backward()
        # 计算多个GPU损失的均值
        loss = multi_gpu.reduce_value(loss, average=True)
        # 计算平均训练损失
        train_loss = (train_loss *i + loss.detach()) / (i + 1)

        # 去每个像素的最大值索引
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        # 获取真实的标签
        true_label = img_label.data.cpu().numpy()

        # 指标计算
        evalution.update(true_label, pre_label)

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    print('Train Loss: ', train_loss.item())

    return train_loss.item()


def evaluate_one_epoch(model, eval_data, evalution, criterion, device):
    model.eval()
    # 重置评价结果
    evalution.reset()
    # 保存验证损失
    eval_loss = torch.zeros(1).to(device)

    prec_time = datetime.now()
    for i, sample in enumerate(eval_data):
        img_data = Variable(sample['img'].to(device))
        img_label = Variable(sample['label'].to(device))

        # 训练
        out = model(img_data)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, img_label)
        # 计算多个GPU损失的均值
        loss = multi_gpu.reduce_value(loss, average=True)
        # 计算平均训练损失
        eval_loss = (eval_loss * i + loss.detach()) / (i + 1)

        # 去每个像素的最大值索引
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        # 获取真实的标签
        true_label = img_label.data.cpu().numpy()

        # 指标计算
        evalution.update(true_label, pre_label)

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    print('Evaluate Loss: ', eval_loss.item())

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)