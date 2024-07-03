# coding=UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from models.network_swinir import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""
modify fast DVD(vedio denoising) 
"""


class swin_Transformer(nn.Module):
    def __init__(self, n_feats, img_size=64, patch_size=4, depths=[6], num_heads=[6],
                 window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(swin_Transformer, self).__init__()

        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.mlp_ratio = mlp_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(1):
            layer = RSTB(dim=n_feats,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(n_feats)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        x = self.conv_after_body(self.forward_features(x)) + x

        return x


# a single branch of proposed SSPSR
class Prior(nn.Module):
    def __init__(self, n_colors, n_feats):
        super(Prior, self).__init__()
        kernel_size = 3
        self.head = nn.Conv2d(64, 128, kernel_size, padding=3 // 2)
        self.feture = nn.Conv2d(128, n_feats, kernel_size, padding=3 // 2)
        self.body = swin_Transformer(n_feats)
        # self.upsample = Upsampler(conv, up_scale, n_feats)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.conv3d1 = wn(nn.Conv3d(1, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)))
        self.conv3d2 = wn(nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))
        # self.conv3d3 = wn(nn.Conv3d(64, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))

        self.tail = nn.Conv2d(n_feats, n_colors, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.head(x)
        y = self.feture(x)
        y = self.body(y)
        y = self.relu(self.tail(y))
        y = y.unsqueeze(1)
        y = self.relu(self.conv3d1(y))
        y = self.conv3d2(y)
        y = y.squeeze(1)

        return y, x


class _3DT_Net(torch.nn.Module):

    def __init__(self, chsi,cmsi, factor, patch_size):
        super(_3DT_Net, self).__init__()  # 调用父类 torch.nn.Module 的构造函数进行初始化。

        self.chsi = chsi  # 输入通道31
        self.cmsi = cmsi
        self.up_factor = factor  # 上采样因子8
        self.patch_size = patch_size  # 图像块大小64
        self.acti = torch.nn.PReLU()  # 参数化的ReLU激活函数
        # 创建可学习的参数，并赋予它们初始值。
        self.delta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(7)])
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.9)) for _ in range(7)])


        self.spatial = Prior(chsi, 180)  # 创建一个名为 Prior 的实例，并将其保存在 self.spatial 变量中。这个实例是用于处理空间信息的先验模型。

        self.fe_conv1 = torch.nn.Conv2d(in_channels=chsi, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv3 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv4 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)

        self.fe_conv5 = torch.nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=3 // 2)
        self.fe_conv6 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv7 = torch.nn.Conv2d(in_channels=448, out_channels=64, kernel_size=3, padding=3 // 2)
        # self.fe_conv8 = torch.nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=3 // 2)
        self.conv_downsample = torch.nn.Conv2d(in_channels=chsi, out_channels=chsi, kernel_size=13, stride=8,
                                               padding=13 // 2)
        self.conv_upsample = torch.nn.ConvTranspose2d(in_channels=chsi, out_channels=chsi, kernel_size=13, stride=8,
                                                      padding=13 // 2)
        self.conv_torgb = torch.nn.Conv2d(in_channels=chsi, out_channels=cmsi, kernel_size=3, stride=1, padding=3 // 2)
        self.conv_tohsi = torch.nn.Conv2d(in_channels=cmsi, out_channels=chsi, kernel_size=3, stride=1, padding=3 // 2)
        self.bicubicsample = torch.nn.Upsample(scale_factor=factor, mode='bicubic', align_corners=False)
        self.reset_parameters()

    def recon_noisy(self, z, noisy, v, RGB, id_layer):
        #    z = self.recon_noisy(z,x,v, RGB,0)
        DELTA = self.delta[id_layer]
        ETA = self.eta[id_layer]

        sz = z.shape
        err1 = RGB - self.conv_torgb(z)  # 将输入的x变成HRMSI与真实的RGB的误差 x:lrhsi上采样变成z
        err1 = self.conv_tohsi(err1)  # 将这个误差3通道转化成31通道
        err2 = noisy - ETA * v  # noisy也就是x ,v是经过特征提取后的向量 两个相减是误差ETA是参数
        err2 = err2.reshape(sz)
        # 它是通过对 z 进行加权和 err1、err2 进行加权后相加得到的。其中 DELTA 和 ETA 是参数，用于控制加权的比例。
        out = (1 - DELTA - DELTA * ETA) * z + DELTA * err1 + DELTA * err2
        return out

    def recon(self, features, recon, HSI, RGB, id_layer):
        DELTA = self.delta[id_layer]
        ETA = self.eta[id_layer]

        sz = recon.shape
        down = self.conv_downsample(recon)
        err1 = self.conv_upsample(down - HSI, output_size=sz)
        to_rgb = self.conv_torgb(recon)
        err_rgb = RGB - to_rgb
        err3 = self.conv_tohsi(err_rgb)
        err3 = err3.reshape(sz)
        ################################################################
        out = (1 - DELTA * ETA) * recon + DELTA * err3 + DELTA * err1 + DELTA * ETA * features
        ################################################################
        return out

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
    def forward(self, HSI, MSI):  # [batch_size ,3 ,7 ,270 ,480] ;
        ## HSI [1 31 6 6]
        ## RGB [1 31 48 48]

        x = self.bicubicsample(HSI)  # 双三次插值上采样生成初始的x
        y = HSI  # LRHSI
        z = x  # HRHSI
        v, fe = self.spatial(self.fe_conv1(z))  # fe是特征，传进去的z做了卷积，v是经过特征提取后的向量 fe:Bx128x64x64 v:Bx31x64x64
        v = v + z
        z = self.recon_noisy(z, x, v, MSI, 0)  # z x v:Bx31x64x64 MSI:Bx3x64x64 z:Bx31x64x64
        conv_out, fe1 = self.spatial(self.fe_conv2(torch.cat((self.fe_conv1(z), fe), 1)))
        conv_out = conv_out + z

        x = self.recon(conv_out, x, y, MSI, id_layer=3)
        z = x
        v, fe2 = self.spatial(self.fe_conv3(torch.cat((self.fe_conv1(z), fe), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, MSI, 0)
        conv_out, fe3 = self.spatial(self.fe_conv4(torch.cat((self.fe_conv1(z), fe2), 1)))
        conv_out = conv_out + z

        x = self.recon(conv_out, x, y, MSI, id_layer=3)
        z = x
        v, fe4 = self.spatial(self.fe_conv5(torch.cat((self.fe_conv1(z), fe, fe2), 1)))
        v = v + z
        z = self.recon_noisy(z, x, v, MSI, 0)
        conv_out, fe5 = self.spatial(self.fe_conv6(torch.cat((self.fe_conv1(z), fe4), 1)))
        conv_out = conv_out + z

        # x = self.recon(conv_out, x, y, MSI,  id_layer=3)
        # z = x
        # v,fe6 = self.spatial(self.fe_conv7(torch.cat((self.fe_conv1(z),fe,fe2,fe4),1)))
        # v=v+z
        # z = self.recon_noisy(z, x, v, MSI, 0)
        # conv_out,_ = self.spatial(self.fe_conv8(torch.cat((self.fe_conv1(z), fe6),1)))
        # conv_out=conv_out+z

        return conv_out


if __name__ == '__main__':
    # model = _3DT_Net(chsi=31, cmsi=3,factor=8, patch_size=8)
    # print(model)
    # HSI = torch.randn((1, 31, 8, 8))
    # MSI = torch.randn((1, 3, 64, 64))
    # x = model(HSI, MSI)
    # print(x.shape)

    # model = _3DT_Net(chsi=93, cmsi=4, factor=8, patch_size=8)
    # print(model)
    # HSI = torch.randn((1, 93, 8, 8))
    # MSI = torch.randn((1, 4, 64, 64))
    # x = model(HSI, MSI)
    # print(x.shape)

    model = _3DT_Net(chsi=191, cmsi=10, factor=8, patch_size=8)
    print(model)
    HSI = torch.randn((1, 191, 8, 8))
    MSI = torch.randn((1, 10, 64, 64))
    x = model(HSI, MSI)
    print(x.shape)

