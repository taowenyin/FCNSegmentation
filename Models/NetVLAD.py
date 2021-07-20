import torch
import torch.nn as nn
import torch.nn.functional as F


# https://www.di.ens.fr/willow/research/netvlad/
# https://zhuanlan.zhihu.com/p/148401141
# https://zhuanlan.zhihu.com/p/148249219
# https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
# https://blog.csdn.net/Yang_137476932/article/details/105169329


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, conv_output_channels=128, alpha=100, normalize_input=True):
        """
        NetVLAD模型实现

        :param num_clusters : int 聚类的数量
        :param conv_output_channels: int CNN输出的特征通道数
        :param alpha: float 初始化参数。参数越大，越难聚类
        :param normalize_input: bool 假如为True，那么使用L2正则化
        """
        super(NetVLAD, self).__init__()

        self.num_clusters = num_clusters
        self.conv_output_channels = conv_output_channels
        self.alpha = alpha
        self.normalize_input = normalize_input

        # NetVLAD中的第一个1x1卷积
        self.conv = nn.Conv2d(conv_output_channels, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, conv_output_channels))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            -1 * self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)


if __name__ == '__main__':
    images = torch.randn(4, 3, 352, 480)

    N, C = images.shape[:2]

    centroids = torch.rand(64, 128)
    data = 2.0 * 100 * centroids
    data = data.unsqueeze(-1).unsqueeze(-1)

    norm = centroids.norm(dim=1)

    print('xxx')