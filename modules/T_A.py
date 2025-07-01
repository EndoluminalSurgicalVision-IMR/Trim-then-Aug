from __future__ import absolute_import, division, print_function

import time
import json
import datasets.lowcam_dataset as datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac
from models.endodac import DPTHead
import numpy as np
import torch.optim as optim
import networks

from utils.utils import *
from utils.layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import random


class Trim:
    @staticmethod
    def patchify(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 32
        assert  imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    @staticmethod
    def patchify_uncertainty(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 32
        assert  imgs.shape[1] % p == 0

        h = imgs.shape[1] // p
        w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p, w, p))
        x = torch.einsum('nhpwq->nhwpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    @staticmethod
    def deal_uncertainty_patch(patch):
        uncertainty_patch = patch    
        uncertainty_mask = torch.ones((patch.shape[0], patch.shape[1]))

        for i in range(uncertainty_patch.shape[0]):
            k = int(0.75 * len(uncertainty_patch[i]))
            largest_k = torch.topk(uncertainty_patch[i], k)
            k_value = largest_k.values
            k_value = k_value[-1]
            for j in range(uncertainty_patch.shape[1]):
                
                if uncertainty_patch[i][j] <= k_value:
                    uncertainty_mask[i][j] = 0
                    
            for j in range(uncertainty_patch.shape[1]):
                if uncertainty_mask[i][j] == 0:
                    seed = random.random() > 0.5
                    if seed:
                        uncertainty_mask[i][j] = 1
                elif uncertainty_mask[i][j] == 1:
                    seed = random.random() > 1 - 0.5 * 0.25 / (1-0.25)
                    if seed:
                        uncertainty_mask[i][j] = 0

        return uncertainty_mask

    @staticmethod
    def unpatchify(x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 32
        h = 8 
        w = 10
        #h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    @staticmethod
    def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device='cuda')  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device='cuda')
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



class Aug:

    def __init__(self):
        pass

    @staticmethod
    def fuse_recon_and_detail(inputs, f_i, models, lam=1.0):

        import torch
        features_recon = models['pretrained_encoder'](inputs["color_aug", f_i, 0])
        recon_outputs = models['pretrained_recon'](features_recon)[('disp',0)]

        transform_input = [inputs["color_aug", f_i, 0], recon_outputs]
        transform_inputs = models["detail_encoder"](torch.cat(transform_input, 1))
        detail_image = models["detail"](transform_inputs)

        residual_image = detail_image[("transform", 0)] + recon_outputs

        return residual_image

    