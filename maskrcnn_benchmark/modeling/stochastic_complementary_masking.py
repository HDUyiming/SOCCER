import random
import warnings

import kornia
import numpy as np
import torch
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data

class Stochastic_Complementary_Masking(nn.Module):
    def __init__(self, block_size, ratio, color_jitter_s, color_jitter_p, blur, mean, std):
        super(Stochastic_Complementary_Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor,labels=None, mask_type='stochastic'):
        #--------------------------------------------------
        # ■ img: Image to be masked
        # ■ label: According to the bounding boxes in the pseudo-label, whether an object is completely 
        #        occluded is determined, and if so, part of the mask is randomly removed.
        # ■ mask_type: we use stochastic masking, but we also provide three types of regular pattern masking 
        #        (This is a supplementary experiment: we employed three types of regular pattern masking:
        #            checkerboard, horizontal stripes, and vertical stripes)
        #--------------------------------------------------

        img = img.clone()
        B, _, H, W = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        if mask_type == 'stochastic':
            mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
            mask = torch.rand(mshape, device=img.device)
            mask = (mask > self.ratio).float()
            mask = resize(mask, size=(H, W))
        
        elif mask_type == 'checkerboard':
            mask = torch.zeros((B, 1, H, W), device=img.device)
            for i in range(0, H, self.block_size * 2):
                for j in range(0, W, self.block_size * 2):
                    mask[:, :, i:i+self.block_size, j:j+self.block_size] = 1
            for i in range(self.block_size, H, self.block_size * 2):
                for j in range(self.block_size, W, self.block_size * 2):
                    mask[:, :, i:i+self.block_size, j:j+self.block_size] = 1

        elif mask_type == 'vertical':
            mask = torch.zeros((B, 1, H, W), device=img.device)
            mask_width = self.block_size
            for i in range(0, W, self.block_size * 2):
                mask[:, :, :, i:i + mask_width] = 1

        elif mask_type == 'horizontal':
            mask = torch.zeros((B, 1, H, W), device=img.device)
            mask_height = self.block_size
            for i in range(0, H, self.block_size * 2):
                mask[:, :, i:i + mask_height, :] = 1


        if labels:
            for label in labels:
                for box in label.bbox:
                    x1, y1, x2, y2 = box.tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                    if mask[:, :,y1:y2+1, x1:x2+1].sum() <1 :

                        box_w = x2 + 1 - x1
                        box_h = y2 + 1 - y1
                        split_y = random.randint(y1 + 2 , y2 + box_h * 5) 
                        split_x = random.randint(x1 + 2 , x2 + box_w * 5)  
                        if split_y < H and split_x < W:
                            mask[:, :,y1:split_y, x1:split_x] = 1 
                        else:                               
                            mask[:, :,y1:y2+1, x1:x2+1] = 1   
            masked_img = img * mask
            complementary_masked_img = img * (1 - mask)
            return masked_img.detach(), complementary_masked_img.detach()
        
        masked_img = img * mask
        complementary_masked_img = img * (1 - mask)
        
        return masked_img.detach(), complementary_masked_img.detach()

