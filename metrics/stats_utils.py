import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


# --------------------------Optimised for Speed
# --------------------------针对速度优化
def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.

    """
    """MoNuSeg 发布的 AJI 版本，没有排列问题，但存在类似 DICE2 的过度惩罚问题。

    快速计算要求实例 ID 是连续排列的，即 [1, 2, 3, 4] 而不是 [2, 3, 6, 10]。
    请事先调用 `remap_label`，`by_size` 标志对结果没有影响。

    """
    true = np.copy(true)  # ? do we need this
    # ? 我们需要这个吗
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    # 预填充值
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    # 缓存配对
    for true_id in true_id_list[1:]:  # 0-th is background
        # 第 0 个是背景
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                # 忽略
                continue  # overlaping background
                # 重叠背景
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # 为每个真实值给出最高 IoU 的预测对，不需要关心
    # about reusing pred instance multiple times
    # 重复使用预测实例多次
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    # 排除那些没有交集的
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    # 打印(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    # 索引到实例 ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    # 将所有未配对的 GT 和预测添加到并集中
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


#####
def get_fast_aji_plus(true, pred):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI 
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.

    """
    """AJI+，一种具有最大唯一配对以获得整体交集的 AJI 版本。
    每个预测实例最多与 1 个 GT 实例配对（1 对 1）映射，不像 AJI
    中一个预测实例可以与多个 GT 实例配对（1 对多）。
    剩余未配对的 GT 和预测实例将被添加到整体并集中。
    1 对 1 的映射防止了 AJI 的过度惩罚。

    快速计算要求实例 ID 是连续排列的，即 [1, 2, 3, 4] 而不是 [2, 3, 6, 10]。
    请事先调用 `remap_label`，`by_size` 标志对结果没有影响。

    """
    true = np.copy(true)  # ? do we need this
    # ? 我们需要这个吗
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    # 预填充值
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    # 缓存配对
    for true_id in true_id_list[1:]:  # 0-th is background
        # 第 0 个是背景
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                # 忽略
                continue  # overlaping background
                # 重叠背景
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    #### Munkres 配对以找到最大唯一配对
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    ### 提取配对成本并移除无效配对
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    # 现在选择所有 IoU != 0.0 的配对，即有交集的
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    # 索引到实例 ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    # 将所有未配对的 GT 和预测添加到并集中
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


#####
def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    """`match_iou` 是 IoU 阈值水平，用于确定 GT 实例 `p` 和预测实例 `g` 之间的配对。
    如果 IoU > `match_iou`，则 `p` 和 `g` 是一对。
    但是，`p` 和 `g` 的配对必须是唯一的（1 个预测实例映射到 1 个 GT 实例）。

    如果 `match_iou` < 0.5，则计算 Munkres 分配（求解二分图中的最小权重匹配）
    以找到最大数量的唯一配对。

    如果 `match_iou` >= 0.5，则所有 IoU(p,g) > 0.5 的配对被证明是唯一的，
    并且配对数量也是最大的。
    
    快速计算要求实例 ID 是连续排列的，
    即 [1, 2, 3, 4] 而不是 [2, 3, 6, 10]。请事先调用 `remap_label`，
    `by_size` 标志对结果没有影响。

    返回:
        [dq, sq, pq]: 测量统计量

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      执行测量的配对信息
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    # 预填充值
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    # 缓存配对 IoU
    for true_id in true_id_list[1:]:  # 0-th is background
        # 第 0 个是背景
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                # 忽略
                continue  # overlaping background
                # 重叠背景
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        # 索引是实例 id - 1
        paired_pred += 1  # hence return back to original
        # 因此返回到原始值
    else:  # * Exhaustive maximal unique pairing
        # * 穷举最大唯一配对
        #### Munkres pairing with scipy library
        #### 使用 scipy 库进行 Munkres 配对
        # the algorithm return (row indices, matched column indices)
        # 算法返回（行索引，匹配的列索引）
        # if there is multiple same cost in a row, index of first occurence
        # 如果一行中有多个相同成本，则返回第一次出现的索引
        # is return, thus the unique pairing is ensure
        # 从而确保唯一配对
        # inverse pair to get high IoU as minimum
        # 反向配对以获得高 IoU 作为最小值
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        ### 提取配对成本并移除无效配对
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # 现在选择那些高于阈值水平的
        # paired with iou = 0.0 i.e no intersection => FP or FN
        # IoU = 0.0 的配对即没有交集 => FP 或 FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    # 获取实际的 FP 和 FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))
    # 打印(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    # 获取 F1 分数即 DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    # 获取 SQ，没有配对的 IoU 为 0，因此没有影响
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#####
def get_fast_dice_2(true, pred):
    """Ensemble dice."""
    """集成 dice。"""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            # 盲目移除背景
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
            # 只是意味着没有背景
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total


#####--------------------------As pseudocode
#####--------------------------作为伪代码
def get_dice_1(true, pred):
    """Traditional dice."""
    """传统 dice。"""
    # cast to binary 1st
    # 首先转换为二值
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


####
def get_dice_2(true, pred):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""
    """计算精准医学挑战赛中使用的集成 Dice。"""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    # 移除背景即 id 0
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / total_markup


#####
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    """重命名所有实例 id，使 id 是连续的，即 [0, 1, 2, 3] 
    而不是 [0, 2, 4, 6]。实例的顺序（哪个在前）会被保留，
    除非 by_size=True，那么实例将被重新排序，
    使较大的细胞核具有较小的 ID。

    参数:
        pred    : 包含实例的 2d 数组，每个实例用非零整数标记
        by_size : 较大的细胞核具有较小的 id（在顶部）

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
        # 没有标签
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        # 按大小降序排序 id
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


#####
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal 
    unique pairing (largest possible match) when pairing points in set B 
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points 
        radius: valid area around a point in setA to consider 
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    """使用 Munkres 或 Kuhn-Munkres 算法找到最优的
    唯一配对（最大可能匹配），当将集合 B 中的点
    与集合 A 中的点配对时，使用距离作为成本函数。

    参数:
        setA, setB: Nx2 大小的 np.array (float32)，包含 N 个不同点的 XY 坐标
        radius: 集合 A 中点周围的有效区域，用于考虑
                集合 B 中的给定坐标作为匹配候选
    返回:
        pairing: pairing 是索引数组
        其中集合 A 中索引 pairing[0] 处的点与
        集合 B 中索引 pairing[1] 处的点配对
        unparedA, unpairedB: 集合 A 和集合 B 中剩余未配对的点

    """
    # * Euclidean distance as the cost matrix
    # * 欧氏距离作为成本矩阵
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # * 使用 scipy 库进行 Munkres 配对
    # the algorithm return (row indices, matched column indices)
    # 算法返回（行索引，匹配的列索引）
    # if there is multiple same cost in a row, index of first occurence 
    # 如果一行中有多个相同成本，则返回第一次出现的索引
    # is return, thus the unique pairing is ensured
    # 从而确保唯一配对
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances 
    # 提取配对成本并移除实例
    # outside of designated radius
    # 在指定半径之外
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:,None], pairedB[:,None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB
