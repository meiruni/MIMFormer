
import logging
# from scipy.ndimage import zoom
from models.IT import InceptionTransformer, default_cfgs
_logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as F


def iformer_small(embed_dims=120,in_chans=64,**kwargs):
    """
    19.866M  4.849G 83.382
    """
    depths = [3]
    embed_dims = [embed_dims]
    num_heads = [3]
    attention_heads = [1]*3 + [3]*3 + [7] * 4 + [9] * 5 + [11] * 3

    model = InceptionTransformer(
        patch_size =16,
        img_size=64,
        depths=depths,
        in_chans = in_chans,
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        **kwargs)
    model.default_cfg = default_cfgs['iformer_small']
    return model
def iformer(patch_size=120,in_chans=64,**kwargs):
    """
    19.866M  4.849G 83.382
    """
    depths = [3]
    patch_size = patch_size
    num_heads = [3]
    attention_heads = [1]*3 + [3]*3 + [7] * 4 + [9] * 5 + [11] * 3

    model = InceptionTransformer(
        patch_size =patch_size,
        img_size=60,
        depths=depths,
        in_chans = in_chans,
        embed_dims=[120],
        num_heads=num_heads,
        attention_heads=attention_heads,
        **kwargs)
    model.default_cfg = default_cfgs['iformer_small']
    return model

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x1)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class Future(nn.Module):
    def __init__(self, n_chans, n_feats):  # 31 180
        super(Future, self).__init__()
        kernel_size = 3
        self.res_blocks = 2

        self.weight_a = nn.Parameter(torch.ones(1).to('cuda:0'))
        self.weight_b = nn.Parameter(torch.ones(1).to('cuda:0'))
        self.weight_c = nn.Parameter(torch.ones(1).to('cuda:0'))
        self.h2 = nn.Conv2d(n_chans, n_feats, kernel_size, padding=3 // 2)
        self.feature = nn.ModuleList()
        for _ in range(self.res_blocks):
            self.feature.append(ResBlock(n_feats, n_feats))
        # self.it= iformer_small(embed_dims=180, in_chans=n_feats)
        self.it1 = iformer(patch_size=3, in_chans=n_feats)
        self.it2 = iformer(patch_size=6, in_chans=n_feats)
        self.it3 = iformer(patch_size=15, in_chans=n_feats)

    def forward(self, x):  # x: Bx64x64x64
        y = self.h2(x)
        for i in range(self.res_blocks):
            y = self.feature[i](y)
        y1 = self.it1(y)
        y2 = self.it2(y)
        y3 = self.it3(y)
        total_weight = self.weight_a + self.weight_b + self.weight_c
        y_body = y1 * (self.weight_a / total_weight) + y2 * (self.weight_b / total_weight) + y3 * (self.weight_c / total_weight)
        # y_body =  self.it(y)
        y = y + y_body
        return y


class MIMFormer(nn.Module):
    def __init__(self,hsi_chans=191, msi_chans=191,scale_factor= 8):
        super(MIMFormer, self).__init__()
        hsi_chans = hsi_chans
        msi_chans = msi_chans
        self.scale_factor = scale_factor
        ####################
        self.MIMFormer = Future(hsi_chans+msi_chans,180)
        self.refine = nn.Sequential(
            nn.Conv2d(180, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, hsi_chans, 3, 1, 1)
        )
    def UP(self,HSI):
        UP_LRHSI= F.interpolate(HSI, scale_factor=self.scale_factor, mode='bicubic')  ### (b N h w)
        return UP_LRHSI


    def forward(self, HSI, MSI):
        ################LR-HSI###################
        UP_LRHSI = self.UP(HSI)
        UP_LRHSI = UP_LRHSI.clamp_(0, 1)
        Data = torch.cat((UP_LRHSI, MSI), 1)
        Data = self.MIMFormer(Data)
        output = self.refine(Data)
        output = output + UP_LRHSI
        output = output.clamp_(0, 1)
        return output




if __name__ == '__main__':
    scale_factor = 16
    #cave pu wdcm

    # LRHSI = torch.randn((2, 31, 8, 8)).to('cuda:0')
    # HRMSI = torch.randn((2, 3, 64, 64)).to('cuda:0')
    # ZY
    LRHSI = torch.randn((2, 76, 20, 20)).to('cuda:0')
    HRMSI = torch.randn((2, 8, 60, 60)).to('cuda:0')

    imgsize = HRMSI.size(2)
    #CAVE
    # model = MainNet(img_size=imgsize,hsi_chans=31,num_feature=34)
    #wdcm
    # model = MainNet(img_size=imgsize, hsi_chans=191, num_feature=201)
    # # #pu
    # model = MIMFormer(hsi_chans=31, msi_chans=3,scale_factor= 8).to('cuda:0')
    # ZY
    model = MIMFormer(hsi_chans=76, msi_chans=8, scale_factor=3).to('cuda:0')
    print(model)
    output = model(LRHSI, HRMSI)
    print(output.shape)
