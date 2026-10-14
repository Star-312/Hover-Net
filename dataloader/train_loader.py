import csv
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """
    """数据加载器。从文件列表加载图像，并使用 albumentation 库进行数据增强。
    数据增强后，生成水平和垂直距离图。

    参数:
        file_list: 要加载的文件名列表
        input_shape: 输入的形状 [h,w] - 在 config.py 中定义
        mask_shape: 输出的形状 [h,w] - 在 config.py 中定义
        mode: 'train'（训练）或 'valid'（验证）
        
    """

    # TODO: doc string
    # TODO: 文档字符串

    def __init__(
        self,
        file_list,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        # 将堆叠的通道拆分为图像和标签
        img = (data[..., :3]).astype("uint8")  # RGB images
        # RGB 图像
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map
        # 实例 ID 图和类型图

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            #type_map[type_map == 5] = 1  # 合并肿瘤性和非肿瘤性
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        # TODO: 记录关于 #input 的硬编码假设
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``0`` -> ``cv2.INTER_NEAREST``（最近邻插值）
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``（双线性插值）
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``（双三次插值）
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``（双三次插值）
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``（双三次插值）
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                # ! 对于 pannuke v0，不进行旋转或平移，仅使用翻转以避免镜像填充
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    # 将图像缩放到其大小的 80-120%，每个轴单独缩放
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    # 平移 -A 到 +A 百分比（每个轴）
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    # 剪切 -5 到 +5 度
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    # 旋转 -179 到 +179 度
                    order=0,  # use nearest neighbour
                    # 使用最近邻插值
                    backend="cv2",  # opencv for fast processing
                    # 使用 opencv 进行快速处理
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # 将位置设为 'center' 进行中心裁剪
                # else 'uniform' for random crop
                # 否则设为 'uniform' 进行随机裁剪
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # 将位置设为 'center' 进行中心裁剪
                # else 'uniform' for random crop
                # 否则设为 'uniform' 进行随机裁剪
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
