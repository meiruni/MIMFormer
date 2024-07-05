import os
import random

from models.Ablation_FSwinU.FFCSWINT import FFCSWINT
from models.Ablation_FSwinU.FFCUNET import FFCUNET
from models.Ablation_FSwinU.SWINU import SWINU
from models.Ablation_MIMFormer.MIMFormer_DCD import MIMFormer_DCD
from models.Ablation_MIMFormer.MIMFormer_NATTEN import MIMFormer_NATTEN
from models.Ablation_MIMFormer.MIMFormer_NDWCONV import MIMFormer_NDWCONV
from models.Ablation_MIMFormer.MIMFormer_NMaxpool import MIMFormer_NMaxpool

from models.MIMFormer_ZY import MIMFormer
from models.FCSwinU import FSwinU
from models.DCT import DCT
from models.network_31 import _3DT_Net
from models.CSSNET import Our_net
from models.SSFCNN import SSFCNN
from models.MSDCNN import MSDCNN
from models.Fusformer import MainNet
from models.TFNet import TFNet

from models.PSRT import PSRTnet
from models.TFNet import ResTFNet
import torch
import numpy as np
from skimage.metrics import structural_similarity
import logging
import argparse


def args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--arch', type=str, default='CSSNET', help=
                                                                    'MIMFormer'
                                                                    'SSFCNN '
                                                                    ' Fusformer'
                                                                    ' MSDCNN'
                                                                    'TFNET'
                                                                    '3DT-Net'
                                                                    'CSSNET'
                                                                    'PSRT'
                                                                    'MIMFormer_DCD'
                                                                    'MIMFormer_NATTEN'
                                                                    'MIMFormer_NMaxpool'
                                                                    'MIMFormer_NDWCONV'
                                                                                        )
    parser.add_argument('--dataset', type=str, default='CAVE', help='ZY1E CAVE  WDCM or PU Harvard')
    parser.add_argument('--upscale_factor', type=int, default=8, help="3 8 16 ")
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--patch_size', type=int, default=8, help='cave wdcm and PU:8  ZY:20')
    parser.add_argument('--num_feature', default=0, type=int, help='hsi+msi  34 201 97')
    parser.add_argument('--img_size', default=64, type=int, help='512  128  128.')
    parser.add_argument('--hsi_chans', type=int, default=0, help='output channel number 31 191 93')
    parser.add_argument('--msi_chans', type=int, default=0, help='3 10 4')

    parser.add_argument('--val_every', type=int, default=5, help='val_every number')
    parser.add_argument('--nEpochs', type=int, default=101, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--save_folder', default='Trained', help='Directory to keep training outputs.')

    parser.add_argument('--outputpath', type=str, default='result', help='Path to output img')
    parser.add_argument('--mode', default=1, type=int, help='Train=1 or Test.')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')

    args = parser.parse_args()
    return args


def select_model(args, device):
    if args.dataset == 'PU':
        args.hsi_chans = 93
        args.msi_chans = 4
        args.num_feature = 97
    elif args.dataset == 'CAVE':
        args.hsi_chans = 31
        args.msi_chans = 3
        args.num_feature = 34
    elif args.dataset == 'WDCM':
        args.hsi_chans = 191
        args.msi_chans = 10
        args.num_feature = 201
    elif args.dataset == 'ZY1E':
        args.hsi_chans = 76
        args.msi_chans = 8
        args.num_feature = 120

    # Build the models
    if args.arch == 'SSFCNN':  # 1
        model = SSFCNN(args.upscale_factor,
                       args.msi_chans,
                       args.hsi_chans).to(device)
    elif args.arch == '3DT-Net':
        model = _3DT_Net(args.hsi_chans,
                         args.msi_chans,
                         args.upscale_factor,
                         args.patch_size).to(device)
    elif args.arch == 'TFNET':
        model = TFNet(args.upscale_factor,
                      args.msi_chans,
                      args.hsi_chans,
                      ).to(device)
    elif args.arch == 'PSRT':
        model = PSRTnet(
            args.hsi_chans,
            args.msi_chans,
            args.upscale_factor,
        ).to(device)
    elif args.arch == 'CSSNET':
        model = Our_net(
                          args.hsi_chans,
                          args.msi_chans,
                          64,
                          args.upscale_factor,
                          ).to(device)
    elif args.arch == 'DCT':
        model = DCT(
            args.hsi_chans,
            args.msi_chans,
            args.upscale_factor,

        ).to(device)
    elif args.arch == 'MSDCNN':
        model = MSDCNN(args.upscale_factor,
                       args.msi_chans,
                       args.hsi_chans).to(device)
    elif args.arch == 'MIMFormer':
        model = MIMFormer(args.hsi_chans,
                          args.msi_chans,
                          args.upscale_factor).to(device)

    elif args.arch == 'MIMFormer_DCD':
        model = MIMFormer_DCD(args.hsi_chans,
                              args.msi_chans,
                              args.upscale_factor).to(device)
    elif args.arch == 'MIMFormer_NATTEN':
        model = MIMFormer_NATTEN(args.hsi_chans,
                                 args.msi_chans,
                                 args.upscale_factor).to(device)
    elif args.arch == 'MIMFormer_NMaxpool':
        model = MIMFormer_NMaxpool(args.hsi_chans,
                                   args.msi_chans,
                                   args.upscale_factor).to(device)
    elif args.arch == 'MIMFormer_NDWCONV':
        model = MIMFormer_NDWCONV(args.hsi_chans,
                                  args.msi_chans,
                                  args.upscale_factor).to(device)
    elif args.arch == 'Fusformer':  # 1
        model = MainNet(args.hsi_chans,
                        args.msi_chans,
                        args.upscale_factor).to(device)
    elif args.arch == 'SWINU':  # 1
        model = SWINU(args.img_size,
                      args.hsi_chans,
                      args.num_feature,
                      args.upscale_factor).to(device)
    elif args.arch == 'FFCUNET':  # 1
        model = FFCUNET(args.hsi_chans,
                        args.num_feature,
                        args.upscale_factor).to(device)
    elif args.arch == 'FFCSWINT':  # 1
        model = FFCSWINT(args.hsi_chans,
                         args.num_feature,
                         args.upscale_factor).to(device)

    else:  # proposed  1
        print('请检查你的模型是否输入正确！！')

    return model


def calc_ergas(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = torch.mean((img_tgt - img_fus) ** 2)
    rmse = rmse ** 0.5
    mean = torch.mean(img_tgt)

    ergas = torch.mean((rmse / mean) ** 2)
    ergas = 100 / 4 * ergas ** 0.5

    return ergas.item()


def calc_psnr(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    mse = torch.mean(torch.square(img_tgt - img_fus))
    img_max = torch.max(img_tgt)
    # img_max = 1.0
    psnr = 10.0 * torch.log10(img_max ** 2 / mse)

    return psnr.item()


def calc_rmse(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = torch.sqrt(torch.mean((img_tgt - img_fus) ** 2))

    return rmse.item()


def calc_sam(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[1], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[1], -1)
    img_tgt = img_tgt / torch.max(img_tgt)
    img_fus = img_fus / torch.max(img_fus)

    A = torch.sqrt(torch.sum(img_tgt ** 2))
    B = torch.sqrt(torch.sum(img_fus ** 2))
    AB = torch.sum(img_tgt * img_fus)

    sam = AB / (A * B)

    sam = torch.arccos(sam)
    sam = torch.mean(sam) * 180 / torch.pi

    return sam.item()


def calc_ssim(img_tgt, img_fus):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt.cpu().numpy()
    img_fus = img_fus.cpu().numpy()

    ssim = structural_similarity(img_tgt, img_fus, data_range=1.0)

    return ssim


class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter(
            '[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")


def create_log_file(file_path):
    file = os.path.exists(file_path)
    if not file:
        try:
            with open(file_path, 'w') as file:
                file.write('This is a log file.')
            print(f"Log file created at: {file_path}")
        except IOError:
            print("Error: Failed to create the log file.")


def load_pretrained_dict(path, opt, model, optimizer):
    load_dict = torch.load(os.path.join(path + '/' + "{}.pth".format(opt.dataset)))
    # 从load_dict中提取需要加载的参数
    load_param = load_dict['param']
    updated_load_param = {}

    for key in load_param:
        if key.startswith("sunet"):
            if "attn_mask" not in key and "relative_position_bias_table" not in key and "relative_position_index" not in key:
                updated_load_param[key] = load_param[key]
    # 加载参数到当前模型
    model_dict = model.state_dict()
    model_dict.update(updated_load_param)
    model.load_state_dict(model_dict)


def checkpoint(epoch, optimizer, opt, model, path):
    model_out_path = path + '/' + "{}.pth".format(opt.dataset)
    save_dict = dict(
        lr=optimizer.state_dict()['param_groups'][0]['lr'],
        param=model.state_dict(),
        adam=optimizer.state_dict(),
        epoch=epoch
    )
    torch.save(save_dict, model_out_path)

    if epoch == opt.nEpochs:
        torch.save(save_dict, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))





