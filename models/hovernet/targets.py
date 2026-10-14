import math
import numpy as np

import torch
import torch.nn.functional as F

from scipy import ndimage
from scipy.ndimage import measurements
from skimage import morphology as morph
import matplotlib.pyplot as plt

from misc.utils import center_pad_to_shape, cropping_center, get_bounding_box
from dataloader.augs import fix_mirror_padding


####
def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    """输入标注必须是原始形状。
    
    该图仅针对裁剪区域内的实例计算，
    但基于原始图像中的原始形状。

    执行以下操作：
    获取每个细胞核实例的水平和垂直距离图。

    """
    orig_ann = ann.copy()  # instance ID map
    # 实例 ID 图
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    # 使用修复后的实例 id 图重新裁剪
    crop_ann = cropping_center(fixed_ann, crop_shape)
    # TODO: deal with 1 label warning
    # TODO: 处理单标签警告
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    # 0 是背景
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # 将边界框扩展 2 像素
        # Because we first pad the ann at line 207, the bboxes
        # 因为我们首先在第 207 行对标注进行了填充，所以边界框
        # will remain valid after expansion
        # 在扩展后仍然有效
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        # 实例质心，四舍五入到最近的像素
        inst_com = list(measurements.center_of_mass(inst_map))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        # 将像素网格的中心移动到实例质心
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        # 移除实例外部的坐标
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        # 将最小值归一化到 -1 范围
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        # 将最大值归一化到 +1 范围
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


####
def gen_targets(ann, crop_shape, **kwargs):
    """Generate the targets for the network."""
    """为网络生成目标。"""
    hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

    hv_map = cropping_center(hv_map, crop_shape)
    np_map = cropping_center(np_map, crop_shape)

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict


####
def prep_sample(data, is_batch=False, **kwargs):
    """
    Designed to process direct output from loader
    """
    """
    用于处理数据加载器的直接输出
    """
    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax, shape):
        ch = np.squeeze(ch.astype("float32"))
        ch = ch / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        # 从 RGBA 热力图中提取 RGB
        ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        ch_cmap = center_pad_to_shape(ch_cmap, shape)
        return ch_cmap

    def prep_one_sample(data):
        shape_array = [np.array(v.shape[:2]) for v in data.values()]
        shape = np.maximum(*shape_array)
        viz_list = []
        viz_list.append(colorize(data["np_map"], 0, 1, shape))
        # map to [0,2] for better visualisation.
        # 映射到 [0,2] 以便更好地可视化。
        # Note, [-1,1] is used for training.
        # 注意，训练时使用的是 [-1,1]。
        viz_list.append(colorize(data["hv_map"][..., 0] + 1, 0, 2, shape))
        viz_list.append(colorize(data["hv_map"][..., 1] + 1, 0, 2, shape))
        img = center_pad_to_shape(data["img"], shape)
        return np.concatenate([img] + viz_list, axis=1)

    # cmap may randomly fails if of other types
    # 如果是其他类型，cmap 可能随机失败
    if is_batch:
        viz_list = []
        data_shape = list(data.values())[0].shape
        for batch_idx in range(data_shape[0]):
            sub_data = {k : v[batch_idx] for k, v in data.items()}
            viz_list.append(prep_one_sample(sub_data))
        return np.concatenate(viz_list, axis=0)
    else:
        return prep_one_sample(data)
