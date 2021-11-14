import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
import skimage.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
from torch.autograd import Variable

class Solver():
    def __init__(self, model, cfg):
        print(torch.cuda.is_available())
        if cfg.scale > 0:
            self.refiner = model()
        else:
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 19499
        
        weights_file = '/scratch/ssrivastava.cse18.iitbhu/CARN/checkpoint/lapcarn/lapcarn_19499.pth'
        self.refiner.load_state_dict(torch.load(weights_file))
        
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, 
                                  device_ids=range(cfg.num_gpu))
        
        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:
                self.refiner.train()

                scale = cfg.scale
                hr_8x, hr_4x, hr_2x, lr = Variable(inputs[0], requires_grad=False), Variable(inputs[1], requires_grad=False), Variable(inputs[2], requires_grad=False), Variable(inputs[3])
                hr_8x = hr_8x.to(self.device)
                hr_4x = hr_4x.to(self.device)
                hr_2x = hr_2x.to(self.device)
                lr = lr.to(self.device)
                
                sr = refiner(lr)
                sr_2x = sr[0]
                sr_4x = sr[1]
                sr_8x = sr[2]
                loss_2x = self.loss_fn(sr_2x, hr_2x)
                loss_4x = self.loss_fn(sr_4x, hr_4x)
                loss_8x = self.loss_fn(sr_8x, hr_8x)
                
                self.optim.zero_grad()
                loss_2x.backward(retain_graph=True)
                loss_4x.backward(retain_graph=True)
                loss_8x.backward()
                torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                
                self.step += 1
                print("[" + str(self.step) + "]", loss_2x.data.item(), loss_4x.data.item(), loss_8x.data.item(), loss_2x.data.item()+loss_4x.data.item()+loss_8x.data.item())
                if cfg.verbose and self.step % cfg.print_interval == 0:
                    psnr_2, psnr_4, psnr_8 = self.evaluate("dataset/Urban100", num_step=self.step)
                    print("[" + str(self.step) + "] PSNR", psnr_2, psnr_4, psnr_8)
                            
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, num_step=0):
        cfg = self.cfg
        mean_psnr_2 = 0
        mean_psnr_4 = 0
        mean_psnr_8 = 0
        self.refiner.eval()
        
        test_data   = TestDataset(test_data_dir)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)
        for step, inputs in enumerate(test_loader):
            lr = inputs[0].squeeze(0)
            hr_2x = inputs[1].squeeze(0)
            hr_4x = inputs[2].squeeze(0)
            hr_8x = inputs[3].squeeze(0)
            

            sr_2x, sr_4x, sr_8x = self.refiner(lr.unsqueeze(0).to(self.device))
            sr_2x = sr_2x.data.squeeze(0)
            sr_4x = sr_4x.data.squeeze(0)
            sr_8x = sr_8x.data.squeeze(0)
            
            hr_2x = hr_2x.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr_2x = sr_2x.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            hr_4x = hr_4x.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr_4x = sr_4x.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            hr_8x = hr_8x.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr_8x = sr_8x.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            mean_psnr_2 += psnr(hr_2x, sr_2x) / len(test_data)
            mean_psnr_4 += psnr(hr_4x, sr_4x) / len(test_data)
            mean_psnr_8 += psnr(hr_8x, sr_8x) / len(test_data)

            '''
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            
            # run refine process in here!
            sr = self.refiner(lr_patch).data
            
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR  
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(test_data)
            '''

        return mean_psnr_2, mean_psnr_4, mean_psnr_8

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = metrics.peak_signal_noise_ratio(im1, im2, data_range=1)
    return psnr
