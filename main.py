from __future__ import print_function
import argparse
import torch

from thop import profile
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket
from Datasets.dataset import *
from thop import clever_format
from utils import *
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

opt = args_parser()

print(opt)
def train(epoch, optimizer):
    epoch_loss = 0
    global current_step
    train_loss = []
    model.train()

    for iteration, batch in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader)):
        # with torch.autograd.set_detect_anomaly(True):
        Z, Y, X = batch[0].float().cuda(), batch[1].float().cuda(), batch[2].float().cuda()

        optimizer.zero_grad()
        # print(Y.shape)
        # print(Z.shape)

        HX = model(Z, Y)

        # alpha = opt.alpha
        loss = criterion(HX, X)
        epoch_loss += loss.detach().cpu().item()

        # tb_logger.add_scalar('total_loss', loss.detach().cpu().item(), current_step)
        current_step += 1

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    logger.info('epoch: ' + str(epoch))
    logger.info('train loss: %.10f' % (np.mean(train_loss)))
    tb_logger.add_scalar('train_loss', np.mean(train_loss), epoch)

    return epoch_loss / len(training_data_loader)


def test(path,save_path):
    load_dict = torch.load(path +'/'+opt.dataset + ".pth")
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])
    psnr = []
    ssim = []
    sam = []
    ergas = []
    rmse = []
    mkdir(opt.outputpath)
    model.eval()
    with torch.no_grad():
        for iteration, batch in tqdm(enumerate(testing_data_loader, 1), total=len(testing_data_loader)):
            # for batch in testing_data_loader:
            Z, Y, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            X = Variable(X).float()
            # print(Y.shape)
            # print(Z.shape)
            HX = model(Z, Y)
            im_name = batch[3][0]
            print(im_name)
            (path, filename) = os.path.split(im_name)
            HX = torch.clamp(HX, 0.0, 1.0)
            psnr.append(calc_psnr(X.detach(), HX.detach()))
            ssim.append(calc_ssim(X.detach(), HX.detach()))
            sam.append(calc_sam(X.detach(), HX.detach()))
            ergas.append(calc_ergas(X.detach(), HX.detach()))
            rmse.append(calc_rmse(X.detach(), HX.detach()))
            mkdir(save_path)
            io.savemat(save_path+'/' + filename, {'HX': torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()})

    psnr_ave = np.mean(psnr)
    ssim_ave = np.mean(ssim)
    sam_ave = np.mean(sam)
    ergas_ave = np.mean(ergas)
    rmse_ave = np.mean(rmse)
    logger.info(
        'Test PSNR (now): %.6f \t SSIM (now): %.6f \t SAM (now): %.6f \t ERGAS (now): %.6f \t RMSE (now): %.6f' % (
            psnr_ave, ssim_ave, sam_ave, ergas_ave, rmse_ave))
    logger.info('Test over.')

    return psnr_ave


def valid(current_epoch=0, max_psnr=0, max_ssim=0, max_psnr_epoch=0, max_ssim_epoch=0):
    logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')
    eval_loss = []
    psnr = []
    ssim = []
    model.eval()
    with torch.no_grad():
        for iteration, batch in tqdm(enumerate(testing_data_loader, 1), total=len(testing_data_loader)):
            # for batch in valid_data_loader:
            Z, Y, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            Y = Variable(Y).float()
            Z = Variable(Z).float()
            X = Variable(X).float()
            # print(Y.shape)
            # print(Z.shape)

            HX = model(Z, Y)

            HX = torch.clamp(HX, 0.0, 1.0)
            loss = criterion(HX, X)
            eval_loss.append(loss.detach().cpu().item())
            psnr.append(calc_psnr(X.detach(), HX.detach()))
            ssim.append(calc_ssim(X.detach(), HX.detach()))

        psnr_ave = np.mean(psnr)
        ssim_ave = np.mean(ssim)

        logger.info('eval loss: %.10f' % (np.mean(eval_loss)))
        logger.info('Eval PSNR (now): %.6f \t SSIM (now): %.6f' % (psnr_ave, ssim_ave))
        if (psnr_ave > max_psnr):
            max_psnr = psnr_ave
            max_psnr_epoch = current_epoch
            checkpoint(epoch,optimizer,opt,model,path)
        if (ssim_ave > max_ssim):
            # print("最大max_ssim%d",max_ssim)
            max_ssim = ssim_ave
            max_ssim_epoch = current_epoch
        logger.info('Eval  PSNR (max): %.6f (%d) \t SSIM (max): %.6f (%d)' % (
            max_psnr, max_psnr_epoch, max_ssim, max_ssim_epoch))
        tb_logger.add_scalar('eval_loss', np.mean(eval_loss), current_epoch)
        tb_logger.add_scalar('psnr', psnr_ave, current_epoch)
        tb_logger.add_scalar('ssim', ssim_ave, current_epoch)
    return max_psnr, max_ssim, max_psnr_epoch, max_ssim_epoch

if __name__ == '__main__':
    set_random_seed(opt.seed)

    #datasets
    test_set = get_test_set(opt.upscale_factor,opt.dataset)
    # test_set = get_patch_test_set(opt.upscale_factor, opt.patch_size,opt.dataset)
    train_set = get_train_set(opt.upscale_factor, opt.patch_size,opt.dataset)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True, pin_memory=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False, pin_memory=True)

    device = 'cuda:0'
    model = select_model(opt,device)
    # model.eval()  # 将模型设置为评估模式
    # LRHSI = torch.randn((1, 31, 8, 8)).to('cuda:0')
    # HRMSI = torch.randn((1, 3, 64, 64)).to('cuda:0')
    # output = model(LRHSI, HRMSI)
    # flops, params = profile(model, inputs=(LRHSI, HRMSI), verbose=False)
    # flops, params = clever_format([flops, params], "%.3f")
    # # 打印FLOPs和参数数量
    # print('FLOPs = ' + str(flops) + ' G')
    # print('Params = ' + str(params) + ' M')
    print(model)
    print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.5)
    path = opt.save_folder +'/'+ opt.arch +'/' + opt.dataset+ str(opt.upscale_factor)
    save_path = opt.outputpath+'/'+ opt.arch +'/' + opt.dataset + str(opt.upscale_factor)
    mkdir(path)
    mkdir(save_path)
    
    criterion = nn.L1Loss()
    log_file_name = path + '/' +'train.log'
    create_log_file(log_file_name)
    logger = Logger(log_file_name=log_file_name, logger_name='train').get_log()
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    tb_logger = SummaryWriter(log_dir='./tf-logs/' + opt.arch +'/'+opt.dataset + '/' + current_time)
    current_step = 0
    
    # train.
    if opt.mode == 1:
        # load_pretrained_dict(path,opt,model,optimizer)
        # load_dict = torch.load(path +'/'+ "{}.pth".format('epoch75WDCM'))
        # opt.lr = load_dict['lr']
        # epoch = load_dict['epoch']
        # model.load_state_dict(load_dict['param'])
        # optimizer.load_state_dict(load_dict['adam'])
        max_psnr = 0
        max_ssim = 0
        max_psnr_epoch = 0
        max_ssim_epoch = 0
        for epoch in range(0, opt.nEpochs):
            avg_loss = train(epoch, optimizer)
            if (epoch % opt.val_every == 0):
                max_psnr, max_ssim, max_psnr_epoch, max_ssim_epoch = valid(epoch, max_psnr, max_ssim, max_psnr_epoch,
                                                                           max_ssim_epoch)
                # checkpoint(epoch,optimizer,opt,model,path)
            torch.cuda.empty_cache()
            scheduler.step()
    else:
    
        SAV_log_file_name = os.path.join(save_path + '/', 'test.log')
        create_log_file(SAV_log_file_name)
        logger = Logger(log_file_name=SAV_log_file_name,
                        logger_name='test').get_log()
        test(path, save_path)
