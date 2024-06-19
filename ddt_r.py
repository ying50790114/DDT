import os
from vgg import *
from PIL import Image
import torchvision.transforms as tvt
import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import cv2
import numpy as np
import torch.nn as nn


class PCAProjectNet(nn.Module):
    def __init__(self):
        super(PCAProjectNet, self).__init__()

    def forward(self, features):  # features: NCWH >> (Numbers, Channels, Weight, High)

        # k is 每個 features 的同一層面積相加 >> k pixels >> k = numbers * weight * high >> a value
        k = features.size(0) * features.size(2) * features.size(3)

        # 對N個features 的對應每個平面算一個平均 >> x_mean ∈ 512-dim >>  (1, 512, 1, 1)
        x_mean  = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        # (Numbers, Channels, Weight, High) 每個平面的每個像素去減相對應channels的平均 (64, 512, 7, 7)
        features = features - x_mean

        # (512, 3136)  << 64*7*7 = 3136
        reshaped_features = features.view(features.size(0), features.size(1), -1).permute(1, 0, 2).contiguous().view(features.size(1), -1)

        # (512, 512)    <<  dimension (512, 3136) * (3136, 512)
        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k

        # 512個 eigenvalues皆二維[ 值, 0.0000 ]   >> eigval ∈ Matrix(512*2) 第一個為最大eigenvalue ; eigvec ∈ Matrix(512*512)
        eigval, eigvec = torch.eig(cov, eigenvectors=True)

        # 取第一個 column 的 eigenvector  (對應最大eigenvalue)   (512, 1)
        first_compo = eigvec[:, 0]

        return first_compo, x_mean



CUDA_VISIBLE_DEVICES="0"

image_trans = tvt.Compose([
    tvt.Resize((224, 224)),
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

data_list = os.listdir('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/single_zeros5x4')

model = vgg16(pretrained=True)

imgs = []
j = 0
for name in data_list:
    j += 1
    img = image_trans(Image.open(os.path.join('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/single_zeros5x4/'+name)).convert('RGB'))
    imgs.append(img.unsqueeze(0))
    if j == 64:
        break
imgs = torch.cat(imgs)      # (64, 3 ,224, 224)

features, _ = model(imgs)   # (64, 512, 7, 7)

pca = PCAProjectNet()

eigenvector, x_mean = pca(features)  # (64, 7, 7)

# test ---------------------------------------------------------------------------------------------------------------------------------------------

test_imgs = image_trans(Image.open(os.path.join('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/single_zeros5x4/8.png')).convert('RGB'))

features_test, _ = model(test_imgs.unsqueeze(0))

features_test = features_test - x_mean      # (1, 512, 7, 7)

test_reshaped_features = features_test.view(features_test.size(0), features_test.size(1), -1).permute(1, 0, 2).contiguous().view(features_test.size(1), -1)

# indicator matrix 7*7  (1, 7, 7)      矩陣相乘 (1, 512) x (512, 49)
projected_map = torch.matmul(eigenvector.unsqueeze(0), test_reshaped_features).view(1, features_test.size(0), -1).view(features_test.size(0), features_test.size(2), features_test.size(3))

maxv = projected_map.max()
minv = projected_map.min()

projected_map *= (maxv + minv) / torch.abs(maxv + minv)

maxv = projected_map.view(projected_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)  # 每張取最大值 (64, 1, 1)

projected_map /= maxv      # (1, 7, 7)  project_map ∈ (0,1)

project_map = F.interpolate(projected_map.unsqueeze(1), size=(imgs.size(2), imgs.size(3)), mode='bilinear', align_corners=False) * 255.   # (1, 1, 224, 224])

img = cv2.resize(cv2.imread(os.path.join('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/single_zeros5x4/8.png')), (224, 224))

mask = project_map[0].repeat(3, 1, 1).permute(1, 2, 0).detach().numpy()  # (224, 224, 3)
mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
save_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)


cv2.imwrite('/home/fagc2267/PycharmProjects/Triplet_loss/DDT_Mnist/5x4_Mnist/test3.jpg', save_img)