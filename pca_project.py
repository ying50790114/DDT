import torch.nn as nn
import torch
import numpy as np


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

        # indicator matrix 7*7  (64, 7, 7)      矩陣相乘 (1, 512) x (512, 3136)
        projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1).view(features.size(0), features.size(2), features.size(3))

        maxv = projected_map.max()
        minv = projected_map.min()

        projected_map *= (maxv + minv) / torch.abs(maxv + minv)

        return projected_map


if __name__ == '__main__':
    img = torch.randn(64, 512, 7, 7)
    pca = PCAProjectNet()
    pca(img)