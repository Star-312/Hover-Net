import math

import cv2
import matplotlib.cm as cm
import numpy as np

from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, map_coordinates

from skimage import morphology as morph

from misc.utils import cropping_center, get_bounding_box


####
def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    
    """
    """处理由于形状增强（缩放、旋转等）过程中插值镜像导致的重复实例问题。
    
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    # 0 是背景
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


####
def gaussian_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply Gaussian blur to input images."""
    """对输入图像应用高斯模糊。"""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    # 默认取输入批次的第一个（我们的情况中始终为 1）
    ksize = random_state.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    ret = ret.astype(np.uint8)
    return [ret]


####
def median_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply median blur to input images."""
    """对输入图像应用中值模糊。"""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    # 默认取输入批次的第一个（我们的情况中始终为 1）
    ksize = random_state.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_hue(images, random_state, parents, hooks, range=None):
    """Perturbe the hue of input images."""
    """扰动输入图像的色相。"""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    # 默认取输入批次的第一个（我们的情况中始终为 1）
    hue = random_state.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        # OpenCV 对 8 位图像使用 0-179 范围
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        # OpenCV 对浮点图像使用 0-360 范围
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_saturation(images, random_state, parents, hooks, range=None):
    """Perturbe the saturation of input images."""
    """扰动输入图像的饱和度。"""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    # 默认取输入批次的第一个（我们的情况中始终为 1）
    value = 1 + random_state.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, random_state, parents, hooks, range=None):
    """Perturbe the contrast of input images."""
    """扰动输入图像的对比度。"""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    # 默认取输入批次的第一个（我们的情况中始终为 1）
    value = random_state.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, random_state, parents, hooks, range=None):
    """Perturbe the brightness of input images."""
    """扰动输入图像的亮度。"""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    # 默认取输入批次的第一个（我们的情况中始终为 1）
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]
