import cv2
import numpy as np

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed
from misc.utils import get_bounding_box, remove_small_objects

import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop


####
def __proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    """使用 XY 坐标图处理细胞核预测。

    参数:
        pred: 预测输出，假设
              通道 0 包含细胞核概率图
              通道 1 包含回归的 X 方向图
              通道 2 包含回归的 Y 方向图

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    # 处理
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already
    # 背景已经是 0

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    ## 细胞核值形成山峰，因此取反得到盆地
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


####
def process(pred_map, nr_types=None, return_centroids=False):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """
    """图像块后处理脚本。

    参数:
        pred_map: tp、np 和 hv 分支的组合输出，顺序相同
        nr_types: nc 分支输出考虑的类型数量
        overlaid_img: 叠加预测实例的图像，`None` 表示不叠加
        type_colour (dict) : `None` 表示使用随机颜色，否则按字典中的颜色叠加各类型实例
        output_dtype: 输出的数据类型
    
    返回:
        pred_inst:     像素级细胞核实例分割预测
        pred_type_out: 像素级细胞核类型预测 

    """
    if nr_types is not None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)
    pred_inst = __proc_np_hv(pred_inst)

    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        # 排除背景
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            # TODO: 更改 bbox 输出格式
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            # * opencv 协议格式可能会出错
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # 少于 3 个点无法构成轮廓，因此跳过，可能也是伪影
            # as the contours obtained via approximation => too small or sthg
            # 因为通过近似获得的轮廓 => 太小或其他问题
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
                # ! 检查异常形状
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            # X 坐标
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            # Y 坐标
            inst_centroid[0] += inst_bbox[0][1]  # X
            # X 坐标
            inst_centroid[1] += inst_bbox[0][0]  # Y
            # Y 坐标
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                # inst_id 应该从 1 开始
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        #### * 获取每个实例 id 的类别，存储在索引 id-1 处
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            # TODO: 重复操作，可能开销较大
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                # ! 如果存在，选择第二多的类别
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # 打印('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # ! 警告：ID 可能不连续
    # inst_id in the dict maps to the same value in the `pred_inst`
    # 字典中的 inst_id 映射到 `pred_inst` 中的相同值
    return pred_inst, inst_info_dict
