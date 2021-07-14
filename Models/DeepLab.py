from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 创建一个深度卷积，其中groups表示有多少个卷积，对于深度卷积来说，卷积的数量与输出的层数相同，而组卷积表示几个层对应一个卷积层
        self.deep_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0,
                              dilation, groups=in_channels, bias=bias)
        # 创建一个逐点卷积
        self.point_wise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                    padding=0, dilation=1, groups=1, bias=bias)

    def fixed_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_input = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))

        return padded_input

    def forward(self, x):
        x = self.fixed_padding(x, self.deep_conv.kernel_size[0], self.deep_conv.dilation[0])
        x = self.deep_conv(x)
        x = self.point_wise(x)
        return x


# 定义一个空洞空间金字塔结构
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, out_stride):
        super(ASPP, self).__init__()

        # 不同out_stride下卷积空洞率
        if out_stride == 8:
            dilations = [1, 6, 12, 18]
        elif out_stride == 16:
            dilations = [1, 12, 24, 36]

        # 定义空洞卷积，当3x3卷积时，padding的大小与空洞的大小相同
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 定义全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 变成 1x2048x1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),  # 输入和输出的层与上面卷积相同
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # ===================================================================================
        # 定义一个1x1的卷积，输入为Concat之后的大小，即4个Conv+1个Pooling输出层数和
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # ===================================================================================

        # 初始化每层的权重
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对卷积核进行权重初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):  # 对BatchNormal层进行权重初始化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        # 使用双线性插值法把x5得到的1x1还原到与x1-x4相同的大小
        x5 = F.interpolate(x5, x4.size()[2:], mode='bilinear', align_corners=True)

        # 使用Concat把数据按照通道数据进行拼接
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # 修改通道数
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# 定义带跳跃的重复块
class Block(nn.Module):
    # repeat_size表示Block中层的重复次数
    def __init__(self, in_channels, out_channels, repeat_size, stride=1, dilation=1, start_with_relu=True, is_last=False):
        super(Block, self).__init__()

        # 通过观察模型流程图，发现只有stride不等于1或输入层数和输出层数不同时，才需要Skip层
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)

        # 最终返回的内容
        rep = []

        # 进Block时第一个可分离卷积，其输入和输出的层不同
        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))
        filters = out_channels

        # 循环添加与上面相同的可分离卷积，其中输入和输出的层相同
        for i in range(repeat_size - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, kernel_size=3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(filters))

        # 如果不以Relu开始，那么第一层就不要
        if not start_with_relu:
            rep = rep[1:]

        # 添加最后一个stride为2的可分离卷积
        if stride != 1:
            rep.append(SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=2))

        # 最后一个Block
        if stride == 1 and is_last:
            rep.append(SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=1))

        # 把rep变成层
        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        x = self.rep(input)

        if self.skip is not None:  # 如果需要有Skip，那么就直接输入
            skip = self.skip(input)
            skip = self.skip_bn(skip)
        else:  # 否则原始数据直接连接
            skip = input

        x += skip

        return x


# 定义改进的Xception网络
class Xception(nn.Module):
    def __init__(self, in_channels=3, out_stride=16):
        super(Xception, self).__init__()

        if out_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilation = (1, 2)
        elif out_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilation = (2, 4)
        else:
            raise NotImplementedError

        # Entry
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, 2, stride=2, start_with_relu=True)
        self.block3 = Block(256, 728, 2, stride=entry_block3_stride, start_with_relu=True)

        # Middle
        self.block4 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block5 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block6 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block7 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block8 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block9 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block10 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block11 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block12 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block13 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block14 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block15 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block16 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block17 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block18 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)
        self.block19 = Block(728, 728, 3, stride=1, dilation=middle_block_dilation, start_with_relu=True)

        # Exit
        self.block20 = Block(728, 1024, repeat_size=2, stride=1, dilation=exit_block_dilation[0], start_with_relu=True, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, kernel_size=3, stride=1, dilation=exit_block_dilation[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 1536, kernel_size=3, stride=1, dilation=exit_block_dilation[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d(1536, 2048, kernel_size=3, stride=1, dilation=exit_block_dilation[1])
        self.bn5 = nn.BatchNorm2d(2048)

        # 初始化每层的权重
        self._init_weight()

    def forward(self, x):
        # Entry
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # 保存1/4大小的数据，与进过ASPP的数据进行融合
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# 定义DeepLab-V3+
class DeepLab(nn.Module):
    def __init__(self, n_classes=21, in_channels=3, out_stride=16, _print=True):
        super(DeepLab, self).__init__()

        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Backbone: Xception")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(out_stride))
            print("Number of Input Channels: {}".format(in_channels))

        # 定义空洞卷积
        self.xception_features = Xception(in_channels, out_stride)

        self.aspp = ASPP(2048, 256, out_stride)

        # ASPP输出要经过1x1的卷积
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Low Level要经过1x1的卷积
        self.conv2 = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1),
        )

    def forward(self, input):
        x, low_level_features = self.xception_features(input)
        x = self.aspp(x)
        x = F.interpolate(x,
                          size=(int(math.ceil(input.size()[-2] / 4)), int(math.ceil(input.size()[-1] / 4))),
                          mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    aspp_model = ASPP(512, 256, 16)
    separable_model = SeparableConv2d(256, 64)
    block_model = Block(64, 128, repeat_size=3, stride=1, is_last=True)
    deeplab_model = DeepLab(3, 21, 16, True)
    aspp_model.eval()
    separable_model.eval()
    block_model.eval()
    deeplab_model.eval()

    image = torch.randn(1, 3, 352, 480)

    # output = aspp_model(image)
    # output = separable_model(output)
    # output = block_model(output)
    output = deeplab_model(image)

    print(output.size())