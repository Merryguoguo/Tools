import os
import torch
import yaml
import numpy as np
from PIL import Image

import torch.nn.functional as F

# 生成用于常规Inpainting任务的随机Mask
# 对于云污染而言，有云检测结果的Mask，一般就不需要生成别的Mask了
def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask

# Mask与Background结合
def mask_image(x, bboxes, config):
    height, width, _ = config['image_shape']
    max_delta_h, max_delta_w = config['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    if x.is_cuda:
        mask = mask.cuda()

    if config['mask_type'] == 'hole':
        result = x * (1. - mask)
    elif config['mask_type'] == 'mosaic':
        # TODO: Matching the mosaic patch size and the mask size
        mosaic_unit_size = config['mosaic_unit_size']
        downsampled_image = F.interpolate(x, scale_factor=1. / mosaic_unit_size, mode='nearest')
        upsampled_image = F.interpolate(downsampled_image, size=(height, width), mode='nearest')
        result = upsampled_image * mask + x * (1. - mask)
    else:
        raise NotImplementedError('Not implemented mask type.')

    return result, mask