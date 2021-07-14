import cfg
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from Models.FCN import FCN
from Models.Seg import Seg
from dataset import CamVidDataset
from torch.utils.data import DataLoader
from PIL import Image


def predict(model, test_data, color_map, out_dir, device):
    net = model.eval()

    for i, sample in enumerate(test_data):
        img_data = sample['img'].to(device)
        img_label = sample['label'].long().to(device)
        file_name = str(sample['file_name'][0])

        out = net(img_data)
        out = F.log_softmax(out, dim=1)

        # 去每个像素的最大值索引
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre = color_map[pre_label][0]
        pre = Image.fromarray(pre)

        if cfg.MODEL_TYPE == cfg.Model.FCN:
            pre.save(out_dir + 'FCN_' + file_name + '_predict.png')
        elif cfg.MODEL_TYPE == cfg.Model.SEG:
            pre.save(out_dir + 'SEG_' + file_name + '_predict.png')
        elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
            pre.save(out_dir + 'DeepLab_' + file_name + '_predict.png')

        print('Done')


if __name__ == '__main__':
    Cam_test = CamVidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = cfg.DATASET[1]

    test_data = DataLoader(Cam_test, batch_size=1, shuffle=True, num_workers=0)

    # 读取标签颜色
    pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    out_dir = "./Results/result_pics/"

    net = None
    if cfg.MODEL_TYPE == cfg.Model.FCN:
        net = FCN(num_classes).to(device)
    elif cfg.MODEL_TYPE == cfg.Model.SEG:
        net = Seg(num_classes).to(device)

    if net is not None:
        if cfg.MODEL_TYPE == cfg.Model.FCN:
            net.load_state_dict(torch.load("./Results/weights/FCN_best.pth"))
        elif cfg.MODEL_TYPE == cfg.Model.SEG:
            net.load_state_dict(torch.load("./Results/weights/SEG_best.pth"))
        elif cfg.MODEL_TYPE == cfg.Model.DEEP_LAB:
            net.load_state_dict(torch.load("./Results/weights/DeepLab_best.pth"))

        predict(net, test_data, cm, out_dir, device)
    else:
        print('Model is None')