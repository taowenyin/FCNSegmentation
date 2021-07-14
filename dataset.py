import os
import pandas as pd
import numpy as np
import torchvision.transforms.functional as ff
import torchvision.transforms as transforms
import torch
import cfg
import os

from PIL import Image
from torch.utils.data import Dataset


class CamVidDataset(Dataset):
    def __init__(self, file_path=None, crop_size=None):
        if len(file_path) != 2:
            raise ValueError('同时需要图片和标签文件路径，图片在前')

        self.img_path = file_path[0]
        self.label_path = file_path[1]

        # 从路径中读取文件，并保存
        self.imgs = self.read_files(self.img_path)
        self.labels = self.read_files(self.label_path)

        # 保存裁剪图片的大小
        self.crop_size = crop_size

        # 标签编码对象
        self.label_processor = LabelProcessor(cfg.class_dict_path)

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        file_name = img.split('/')[-1].split('.')[0]

        # 打开图片
        img = Image.open(img)  # 480 360
        label = Image.open(label).convert('RGB')

        # 裁剪图片
        img, label = self.center_crop(img, label, self.crop_size)
        # 对图像进行标准化，对标签进行编码
        img, label = self.image_transform(img, label)

        sample = {'img': img, 'label': label, 'file_name': file_name}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_files(self, path):
        files_list = os.listdir(path)
        files_path_list = [os.path.join(path, img) for img in files_list]
        files_path_list.sort()

        return files_path_list

    def center_crop(self, image, label, crop_size):
        # 对图片进行裁剪
        image = ff.center_crop(image, crop_size)
        label = ff.center_crop(label, crop_size)

        return image, label

    def image_transform(self, image, label):
        # 图片转化为Tensor和标准化的流程
        transform_img = transforms.Compose(
            [
                # 把图像转化为Tensor
                transforms.ToTensor(),
                # 把图像进行归一化
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

        # 对所有图像进行转化
        img = transform_img(image)
        # 对标签进行编码
        label = self.label_processor.encode_label_img(label)
        label = torch.from_numpy(label)

        return img, label


class LabelProcessor:
    def __init__(self, file_path):
        # 从文件当中读取每个类别的像素值
        self.color_map = self.read_color_map(file_path)

        # 通过colr_map对标签进行一一对应
        self.cm2lbl = self.encode_label_pix(self.color_map)

    # 载入颜色映射
    def read_color_map(self, file_path):
        # 读取color map的CSV文件
        pd_label_color = pd.read_csv(file_path, sep=',')
        color_map = []
        # 循环读取每行的颜色，并保存到color_map中
        for i in range(len(pd_label_color.index)):
            # 读取一行数据
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            color_map.append(color)

        return color_map

    # 把颜色和类别进行对应
    def encode_label_pix(self, color_map):
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(color_map):
            # 通过哈希函数定义哈希映射，提高查找效率
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        return cm2lbl

    def encode_label_img(self, label):
        data = np.array(label, dtype='int32')
        # 计算诶个像素的哈希索引
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]

        a = np.array(self.cm2lbl[idx], dtype='int64')

        # 通过哈希表查询把标签图中的像素变为每个像素的分类
        return np.array(self.cm2lbl[idx], dtype='int64')
