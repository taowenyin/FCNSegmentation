from Models.DeepLab import DeepLab
from torchvision.models.segmentation import deeplabv3_resnet101

if __name__ == '__main__':
     deeplab = DeepLab(n_classes=12)
     deeplabv3 = deeplabv3_resnet101(num_classes=12)

     print(deeplab)
     print('======================================================')
     print(deeplabv3)