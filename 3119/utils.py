from collections.abc import Sequence
import gzip
import hashlib
import os
import os.path as osp
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile

from PIL import Image
import mmcv
import numpy as np
import torch
import torchvision
from typing import Tuple
import torch.nn as nn

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(torch.float32)
    elif isinstance(data, Image.Image):
        return torchvision.transforms.functional.to_tensor(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')



def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image.Image):
        data = np.array(data, dtype=np.uint8)
        if data.ndim < 3:
            data = np.expand_dims(data, axis=-1)
        data = np.rollaxis(data, 2)  # HWC to CHW
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to numpy.')



class BlockwiseMaskGenerator(object):
    """Generate random block for the image.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(self,
                 input_size=192,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.6,
                 mask_only=False,
                 mask_color='zero',
                ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_only = mask_only
        self.mask_color = mask_color
        assert self.mask_color in ['mean', 'zero', 'rand','learnable']
        if self.mask_color != 'zero':
            assert mask_only == False

        if self.mask_color == 'learnable':
            self.learnable_color = nn.Parameter(torch.rand(3))

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img=None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.from_numpy(mask)  # [H, W]

        if self.mask_color == 'mean':
            if isinstance(img, Image.Image):
                img = np.array(img)
                mask_ = to_numpy(mask).reshape((self.rand_size * self.scale, -1, 1))
                mask_ = mask_.repeat(
                    self.model_patch_size, axis=0).repeat(self.model_patch_size, axis=1)
                mean = img.reshape(-1, img.shape[2]).mean(axis=0)
                img = np.where(mask_ == 1, img, mean)
                img = Image.fromarray(img.astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                mask_ = to_tensor(mask)
                mask_ = mask_.repeat_interleave(self.model_patch_size, 0).repeat_interleave(
                    self.model_patch_size, 1).contiguous()
                img = img.clone()
                mean = img.mean(dim=[1,2])
                for i in range(img.size(0)):
                    img[i, mask_ == 1] = mean[i]

        if self.mask_color =="learnable":
            if isinstance(img, Image.Image):
                img = np.array(img)
                mask_ = to_numpy(mask).reshape((self.rand_size * self.scale, -1, 1))
                mask_ = mask_.repeat(
                    self.model_patch_size, axis=0).repeat(self.model_patch_size, axis=1)
                mean_mask = img.reshape(-1, img.shape[2]).mean(axis=0)

                img = np.where(mask_ == 1, img, mean)
                img = Image.fromarray(img.astype(np.uint8))

            elif isinstance(img, torch.Tensor):
                mask_ = to_tensor(mask)
                mask_ = mask_.repeat_interleave(self.model_patch_size, 0).repeat_interleave(
                    self.model_patch_size, 1).contiguous()
                img = img.clone()
                # 生成缩略图
                thumbnail =torch.nn.functional.interpolate(img.unsqueeze(0), size=(14, 14), mode='bilinear')

                # 将缩略图扩展为原始大小
                resized_thumbnail = thumbnail.repeat(1, 1, 16, 16)
                resized_thumbnail = resized_thumbnail.squeeze()
                # mask_ 是一个 224x224 的 mask，其中 0 表示某种像素位置，1 表示另一种像素位置
                # 将 mask_ 扩展为 3D 的形状，与 resized_thumbnail 保持一致
                mask_3d = mask_.unsqueeze(0).expand_as(resized_thumbnail)
                # 将 mask_ 中为 0 的位置保持原始值，为 1 的位置替换为对应缩略图的像素值
                result = img.clone()
                result[mask_3d == 1] = resized_thumbnail[mask_3d == 1]
                # result_numpy = result.cpu().numpy().transpose(1, 2, 0)  # 将通道维度移到最后一个维度上
                # import matplotlib.pyplot as plt
                # # 显示图像
                # plt.imshow(result_numpy)
                # plt.axis('off')
                # plt.show()

                img = result
                # mean = img.mean(dim=[1, 2])
                # for i in range(img.size(0)):
                #     img[i, mask_ == 1] = mean[i]

        if self.mask_only:
            return mask
        else:
            return img, mask

