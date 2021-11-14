import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(hr_8x, hr_4x, hr_2x, lr, size):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize_2 = size*2
    hsize_4 = size*4
    hsize_8 = size*8
    hx_2, hy_2 = x*2, y*2
    hx_4, hy_4 = x*4, y*4
    hx_8, hy_8 = x*8, y*8

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr_8x = hr_8x[hy_8:hy_8+hsize_8, hx_8:hx_8+hsize_8].copy()
    crop_hr_4x = hr_4x[hy_4:hy_4+hsize_4, hx_4:hx_4+hsize_4].copy()
    crop_hr_2x = hr_2x[hy_2:hy_2+hsize_2, hx_2:hx_2+hsize_2].copy()

    return crop_hr_8x, crop_hr_4x, crop_hr_2x, crop_lr


def random_flip_and_rotate(im1, im2, im3, im4):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)
        im3 = np.flipud(im3)
        im4 = np.flipud(im4)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)
        im3 = np.fliplr(im3)
        im4 = np.fliplr(im4)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)
    im3 = np.rot90(im3, angle)
    im4 = np.rot90(im4, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy(), im3.copy(), im4.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")
        
        self.lr = [v[:] for v in h5f["X8"].values()]
        self.hr_2x = [v[:] for v in h5f["X4"].values()]
        self.hr_4x = [v[:] for v in h5f["X2"].values()]
        self.hr_8x = [v[:] for v in h5f["HR"].values()]
        
        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size

        item = [self.hr_8x[index], self.hr_4x[index], self.hr_2x[index], self.lr[index]]
        item = random_crop(item[0], item[1], item[2], item[3], size)
        item = random_flip_and_rotate(item[0], item[1], item[2], item[3])
        
        return [self.transform(item[0]), self.transform(item[1]), self.transform(item[2]), self.transform(item[3])]

    def __len__(self):
        return len(self.hr_8x)
        

class TestDataset(data.Dataset):
    def __init__(self, dirname):
        super(TestDataset, self).__init__()

        all_files = glob.glob(os.path.join(dirname, "*.png"))
        self.lr = [name for name in all_files if "lr" in name]
        self.hr_2x = [name for name in all_files if "2x" in name]
        self.hr_4x = [name for name in all_files if "4x" in name]
        self.hr_8x = [name for name in all_files if "8x" in name]

        self.lr.sort()
        self.hr_2x.sort()
        self.hr_4x.sort()
        self.hr_8x.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        lr = Image.open(self.lr[index])
        hr_2x = Image.open(self.hr_2x[index])
        hr_4x = Image.open(self.hr_4x[index])
        hr_8x = Image.open(self.hr_8x[index])

        lr = lr.convert("RGB")
        hr_2x = hr_2x.convert("RGB")
        hr_4x = hr_4x.convert("RGB")
        hr_8x = hr_8x.convert("RGB")

        return self.transform(lr), self.transform(hr_2x), self.transform(hr_4x), self.transform(hr_8x)

    def __len__(self):
        return len(self.hr_8x)
