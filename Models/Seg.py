import torch
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torch import nn


class Seg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 创建VGG模型对象，并使用ImageNet预训练权重
        pretrained_net = models.vgg16(pretrained=True)

        # 设置Pooling层返回pooling的索引
        pool_index = [4, 9, 16, 23, 30]
        for index in pool_index:
            pretrained_net.features[index].return_indices = True

        # 编码层
        self.encoder_1 = pretrained_net.features[:4]
        self.pool_1 = pretrained_net.features[4]

        self.encoder_2 = pretrained_net.features[5:9]
        self.pool_2 = pretrained_net.features[9]

        self.encoder_3 = pretrained_net.features[10:16]
        self.pool_3 = pretrained_net.features[16]

        self.encoder_4 = pretrained_net.features[17:23]
        self.pool_4 = pretrained_net.features[23]

        self.encoder_5 = pretrained_net.features[24:30]
        self.pool_5 = pretrained_net.features[30]

        # 解码层
        # 上采样
        self.uppool_5 = nn.MaxUnpool2d(2, 2)
        self.decoder_5 = self.decoder(512, 512)

        self.uppool_4 = nn.MaxUnpool2d(2, 2)
        self.decoder_4 = self.decoder(512, 256)

        self.uppool_3 = nn.MaxUnpool2d(2, 2)
        self.decoder_3 = self.decoder(256, 128)

        self.uppool_2 = nn.MaxUnpool2d(2, 2)
        self.decoder_2 = self.decoder(128, 64, 2)

        self.uppool_1 = nn.MaxUnpool2d(2, 2)
        self.decoder_1 = self.decoder(64, num_classes, 2)

    def forward(self, x):
        encoder_1 = self.encoder_1(x)
        encoder_1_size = encoder_1.size()
        pool_1, indices_1 = self.pool_1(encoder_1)

        encoder_2 = self.encoder_2(pool_1)
        encoder_2_size = encoder_2.size()
        pool_2, indices_2 = self.pool_2(encoder_2)

        encoder_3 = self.encoder_3(pool_2)
        encoder_3_size = encoder_3.size()
        pool_3, indices_3 = self.pool_3(encoder_3)

        encoder_4 = self.encoder_4(pool_3)
        encoder_4_size = encoder_4.size()
        pool_4, indices_4 = self.pool_4(encoder_4)

        encoder_5 = self.encoder_5(pool_4)
        encoder_5_size = encoder_5.size()
        pool_5, indices_5 = self.pool_5(encoder_5)

        uppool_5 = self.uppool_5(input=pool_5, indices=indices_5, output_size=encoder_5_size)
        decoder_5 = self.decoder_5(uppool_5)

        uppool_4 = self.uppool_4(input=decoder_5, indices=indices_4, output_size=encoder_4_size)
        decoder_4 = self.decoder_4(uppool_4)

        uppool_3 = self.uppool_3(input=decoder_4, indices=indices_3, output_size=encoder_3_size)
        decoder_3 = self.decoder_3(uppool_3)

        uppool_2 = self.uppool_2(input=decoder_3, indices=indices_2, output_size=encoder_2_size)
        decoder_2 = self.decoder_2(uppool_2)

        uppool_1 = self.uppool_1(input=decoder_2, indices=indices_1, output_size=encoder_1_size)
        decoder_1 = self.decoder_1(uppool_1)

        return decoder_1

    # 获取解码层
    def decoder(self, input_channel, output_channel, num=3):
        # 解码层
        decoder_layer = None
        if num == 3:
            decoder_layer = nn.Sequential(
                nn.Conv2d(input_channel, input_channel, 3, padding=1),
                nn.Conv2d(input_channel, input_channel, 3, padding=1),
                nn.Conv2d(input_channel, output_channel, 3, padding=1)
            )
        elif num == 2:
            decoder_layer = nn.Sequential(
                nn.Conv2d(input_channel, input_channel, 3, padding=1),
                nn.Conv2d(input_channel, output_channel, 3, padding=1)
            )

        return decoder_layer
