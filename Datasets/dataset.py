# -*- coding: UTF-8 -*-

from os.path import exists, join, basename
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
import scipy.io as sio
import random
import torch

def normalize(data):
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    data = data.reshape((h, w, c))
    return data

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".mat"])


def load_img(filepath):
    x = sio.loadmat(filepath)
    x = x['hrhsi']
    x = x.astype(np.float32)
    x = torch.tensor(x).float()
    return x


def load_img1(filepath):
    x = sio.loadmat(filepath)
    x = x['hrmsi']
    x = x.astype(np.float32)
    x = torch.tensor(x).float()
    return x


def load_img2(filepath):
    x = sio.loadmat(filepath)
    x = x['lrhsi']
    x = x.astype(np.float32)
    x = torch.tensor(x).float()
    return x

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def input_transform():
    return Compose([
        ToTensor(),
    ])


#PRE
def get_train_set(upscale_factor, patch_size,dataset):
    root_dir = "./Datasets/" + dataset
    train_dir1 = root_dir + "/train/X"
    train_dir2 = root_dir + "/train/Y"
    train_dir3 = root_dir + "/train/Z_"+str(upscale_factor)
    dataset_train = DatasetFromFolder_patch_train(train_dir1,train_dir2,train_dir3, upscale_factor,patch_size,input_transform=input_transform())
    return dataset_train

def get_patch_test_set(upscale_factor, patch_size,dataset):
    root_dir = "./Datasets/" + dataset
    test_dir1 = root_dir + "/test/X"
    test_dir2 = root_dir + "/test/Y"
    test_dir3 = root_dir + "/test/Z_"+str(upscale_factor)
    dataset_test = DatasetFromFolder_patch_test(test_dir1,test_dir2,test_dir3,  upscale_factor,16,input_transform=(input_transform))

    return dataset_test


def get_test_set(upscale_factor,dataset):
    root_dir = "./Datasets/" + dataset
    test_dir1 = root_dir + "/test/X"
    test_dir2 = root_dir + "/test/Y"
    test_dir3 = root_dir + "/test/Z_"+str(upscale_factor)
    dataset_test = DatasetFromFolder_test(test_dir1,test_dir2,test_dir3,input_transform=input_transform())
    return dataset_test

class DatasetFromFolder_patch_train(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3, upscale_factor, patch_size, input_transform=None):
        super(DatasetFromFolder_patch_train, self).__init__()

        self.patch_size = patch_size
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)

        self.lens = 20000
        # self.lens = 160

        self.xs = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.x_blurs = []
        for img in self.image_filenames3:
            self.x_blurs.append(load_img2(img))

        self.upscale_factor = upscale_factor
        self.input_transform = input_transform

    def __getitem__(self, index):
        ind = index % len(self.image_filenames1)
        # ind =0
        img = self.xs[ind]
        img2 = self.ys[ind]
        img3 = self.x_blurs[ind]
        upscale_factor = self.upscale_factor
        w = np.random.randint(0, img3.shape[0] -self.patch_size)
        h = np.random.randint(0, img3.shape[1] -self.patch_size)
        X = img[w*upscale_factor:(w + self.patch_size)*upscale_factor, h*upscale_factor:(h + self.patch_size)*upscale_factor, :]
        Y = img2[w*upscale_factor:(w + self.patch_size)*upscale_factor, h*upscale_factor:(h + self.patch_size)*upscale_factor, :]
        Z = img3[w:w + self.patch_size, h:h + self.patch_size, :]

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        X = torch.rot90(X, rotTimes, [0, 1])
        Y = torch.rot90(Y, rotTimes, [0, 1])
        Z = torch.rot90(Z, rotTimes, [0, 1])

        # Random vertical Flip
        for j in range(vFlip):
            X = X.flip(1)
            Y = Y.flip(1)
            Z = Z.flip(1)

        # Random Horizontal Flip
        for j in range(hFlip):
            X = X.flip(0)
            Y = Y.flip(0)
            Z = Z.flip(0)

        X = X.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)

        return Z, Y, X

    def __len__(self):
        return self.lens


# patch_test
class DatasetFromFolder_patch_test(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3, upscale_factor, patch_size, input_transform=None):
        super(DatasetFromFolder_patch_test, self).__init__()

        self.patch_size = patch_size
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)

        self.xs = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.x_blurs = []
        for img in self.image_filenames3:
            self.x_blurs.append(load_img2(img))

        self.upscale_factor = upscale_factor
        self.input_transform = input_transform

    def __getitem__(self, index):
        ind = index % len(self.image_filenames1)
        img = self.xs[ind]
        img2 = self.ys[ind]
        img3 = self.x_blurs[ind]
        upscale_factor = self.upscale_factor
        w=0
        h=0
        X = img[w*upscale_factor:(w + self.patch_size)*upscale_factor, h*upscale_factor:(h + self.patch_size)*upscale_factor, :]
        Y = img2[w*upscale_factor:(w + self.patch_size)*upscale_factor, h*upscale_factor:(h + self.patch_size)*upscale_factor, :]
        Z = img3[w:w + self.patch_size, h:h + self.patch_size, :]

        X = X.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)

        return Z, Y, X

    # def __len__(self):
    #     return self.lens
    def __len__(self):
        return min(len(self.xs), len(self.ys), len(self.x_blurs))



#test

class DatasetFromFolder_test(data.Dataset):
    def __init__(self, image_dir1, image_dir2, image_dir3, input_transform=None):
        super(DatasetFromFolder_test, self).__init__()
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.image_filenames3 = [join(image_dir3, x) for x in listdir(image_dir3) if is_image_file(x)]
        self.image_filenames1 = sorted(self.image_filenames1)
        self.image_filenames2 = sorted(self.image_filenames2)
        self.image_filenames3 = sorted(self.image_filenames3)
        # self.upscale_factor = upscale_factor
        self.input_transform = input_transform

        self.xs = []
        self.xs_name = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))
            self.xs_name.append(img)

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.zs = []
        for img in self.image_filenames3:
            self.zs.append(load_img2(img))

    def __getitem__(self, index):
        X = self.xs[index]
        Y = self.ys[index]
        Z = self.zs[index]

        X = X.permute(2, 0, 1)
        # Y = Y.permute(2, 0, 1)
        Z = Z.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)

        return Z, Y, X, self.xs_name[index]

    def __len__(self):
        return len(self.image_filenames1)


