import torch
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torch import nn


class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 创建VGG模型对象，并使用ImageNet预训练权重
        pretrained_net = models.vgg16_bn(pretrained=True)

        # 定义5个Pooling过程
        self.stage1 = pretrained_net.features[:7]
        self.stage2 = pretrained_net.features[7:14]
        self.stage3 = pretrained_net.features[14:24]
        self.stage4 = pretrained_net.features[24:34]
        self.stage5 = pretrained_net.features[34:]

        # FCN-32s中的卷积核
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        # 初始化卷基层参数为0
        # self.scores1.weight = nn.Parameter(torch.tensor(np.zeros(self.scores1.weight.shape),
        #                                                 dtype=torch.float32, requires_grad=True))
        # FCN-16s中的卷积核
        self.scores2 = nn.Conv2d(512, num_classes, 1)
        # self.scores2.weight = nn.Parameter(torch.tensor(np.zeros(self.scores2.weight.shape),
        #                                                 dtype=torch.float32, requires_grad=True))
        # FCN-8s中的卷积核
        self.scores3 = nn.Conv2d(256, num_classes, 1)
        # self.scores3.weight = nn.Parameter(torch.tensor(np.zeros(self.scores3.weight.shape),
        #                                                 dtype=torch.float32, requires_grad=True))

        # 过渡卷积
        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

        # 反卷积，进行8倍还原
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        # 初始化权重
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes, num_classes, 16)

        # 2倍扩大,(4, 2, 1)
        self.upsample_2x_1 = nn.ConvTranspose2d(12, 12, 4, 2, 1, bias=False)
        self.upsample_2x_1.weight.data = self.bilinear_kernel(12, 12, 4)

        self.upsample_2x_2 = nn.ConvTranspose2d(12, 12, 4, 2, 1, bias=False)
        self.upsample_2x_2.weight.data = self.bilinear_kernel(12, 12, 4)

    def forward(self, x):  # 3 480 352  B C W H
        s1 = self.stage1(x)  # 64 176 240
        s2 = self.stage2(s1)  # 128 88 120
        s3 = self.stage3(s2)  # 256 44 60
        s4 = self.stage4(s3)  # 512 22 30
        s5 = self.stage5(s4)  # 512 11 15

        scores1 = self.scores1(s5)  # 12 11 15
        s5 = self.upsample_2x_1(scores1)  # 12 22 30

        s4 = self.scores2(s4)  # 12 22 30
        add1 = s5 + s4  # 12 22 30

        add1 = self.upsample_2x_2(add1)  # 12 44 60

        s3 = self.scores3(s3)

        add2 = add1 + s3

        output = self.upsample_8x(add2)  # 12 352 480

        return output

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
        weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
        return torch.from_numpy(weight)
