import multiprocessing as mp
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from multiprocessing import Lock, Pool

mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging
# ! 必须放在顶部以便 VScode 调试

import argparse
import glob
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import time
from functools import reduce
from importlib import import_module

import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import SerializeArray, SerializeFileList
from docopt import docopt
from misc.utils import (
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.wsi_handler import get_file_handler

from . import base

thread_lock = Lock()


####
def _init_worker_child(lock_):
    global lock
    lock = lock_


####
def _remove_inst(inst_map, remove_id_list):
    """Remove instances with id in remove_id_list.
    """移除 id 在 remove_id_list 中的实例。
    
    Args:
    参数:
        inst_map: map of instances
        inst_map: 实例图
        remove_id_list: list of ids to remove from inst_map
        remove_id_list: 要从实例图中移除的 id 列表
    """
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map


####
def _get_patch_top_left_info(img_shape, input_size, output_size):
    """Get top left coordinate information of patches from original image.
    """从原始图像获取补丁的左上角坐标信息。

    Args:
    参数:
        img_shape: input image shape
        img_shape: 输入图像形状
        input_size: patch input shape
        input_size: 补丁输入形状
        output_size: patch output shape
        output_size: 补丁输出形状

    """
    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    # 从原始图像生成子补丁索引
    output_tl_y_list = np.arange(
        in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32
    )
    output_tl_x_list = np.arange(
        in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32
    )
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack(
        [output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1
    )
    input_tl = output_tl - in_out_diff // 2
    return input_tl, output_tl


#### all must be np.array
#### 所有必须是 np.array
def _get_tile_info(img_shape, tile_shape, ambiguous_size=128):
    """Get information of tiles used for post processing.
    """获取用于后处理的切片信息。

    Args:
    参数:
        img_shape: input image shape
        img_shape: 输入图像形状
        tile_shape: tile shape used for post processing
        tile_shape: 用于后处理的切片形状
        ambiguous_size: used to define area at tile boundaries
        ambiguous_size: 用于定义切片边界区域
    
    """
    # * get normal tiling set
    # * 获取正常切片集
    tile_grid_top_left, _ = _get_patch_top_left_info(img_shape, tile_shape, tile_shape)
    tile_grid_bot_right = []
    for idx in list(range(tile_grid_top_left.shape[0])):
        tile_tl = tile_grid_top_left[idx][:2]
        tile_br = tile_tl + tile_shape
        axis_sel = tile_br > img_shape
        tile_br[axis_sel] = img_shape[axis_sel]
        tile_grid_bot_right.append(tile_br)
    tile_grid_bot_right = np.array(tile_grid_bot_right)
    tile_grid = np.stack([tile_grid_top_left, tile_grid_bot_right], axis=1)
    tile_grid_x = np.unique(tile_grid_top_left[:, 1])
    tile_grid_y = np.unique(tile_grid_top_left[:, 0])
    # * get tiling set to fix vertical and horizontal boundary between tiles
    # * 获取切片集以修复切片之间的垂直和水平边界
    # for sanity, expand at boundary `ambiguous_size` to both side vertical and horizontal
    # 为了完整性，在边界处将 `ambiguous_size` 向两侧垂直和水平扩展
    stack_coord = lambda x: np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    tile_boundary_x_top_left = np.meshgrid(
        tile_grid_y, tile_grid_x[1:] - ambiguous_size
    )
    tile_boundary_x_bot_right = np.meshgrid(
        tile_grid_y + tile_shape[0], tile_grid_x[1:] + ambiguous_size
    )
    tile_boundary_x_top_left = stack_coord(tile_boundary_x_top_left)
    tile_boundary_x_bot_right = stack_coord(tile_boundary_x_bot_right)
    tile_boundary_x = np.stack(
        [tile_boundary_x_top_left, tile_boundary_x_bot_right], axis=1
    )
    #
    tile_boundary_y_top_left = np.meshgrid(
        tile_grid_y[1:] - ambiguous_size, tile_grid_x
    )
    tile_boundary_y_bot_right = np.meshgrid(
        tile_grid_y[1:] + ambiguous_size, tile_grid_x + tile_shape[1]
    )
    tile_boundary_y_top_left = stack_coord(tile_boundary_y_top_left)
    tile_boundary_y_bot_right = stack_coord(tile_boundary_y_bot_right)
    tile_boundary_y = np.stack(
        [tile_boundary_y_top_left, tile_boundary_y_bot_right], axis=1
    )
    tile_boundary = np.concatenate([tile_boundary_x, tile_boundary_y], axis=0)
    # * get tiling set to fix the intersection of 4 tiles
    # * 获取切片集以修复 4 个切片的交叉点
    tile_cross_top_left = np.meshgrid(
        tile_grid_y[1:] - 2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size
    )
    tile_cross_bot_right = np.meshgrid(
        tile_grid_y[1:] + 2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size
    )
    tile_cross_top_left = stack_coord(tile_cross_top_left)
    tile_cross_bot_right = stack_coord(tile_cross_bot_right)
    tile_cross = np.stack([tile_cross_top_left, tile_cross_bot_right], axis=1)
    return tile_grid, tile_boundary, tile_cross


####
def _get_chunk_patch_info(
    img_shape, chunk_input_shape, patch_input_shape, patch_output_shape
):
    """Get chunk patch info. Here, chunk refers to tiles used during inference.
    """获取块补丁信息。这里，块指的是推理期间使用的切片。

    Args:
    参数:
        img_shape: input image shape
        img_shape: 输入图像形状
        chunk_input_shape: shape of tiles used for post processing
        chunk_input_shape: 用于后处理的切片形状
        patch_input_shape: input patch shape
        patch_output_shape: output patch shape

    """
    round_to_multiple = lambda x, y: np.floor(x / y) * y
    patch_diff_shape = patch_input_shape - patch_output_shape

    chunk_output_shape = chunk_input_shape - patch_diff_shape
    chunk_output_shape = round_to_multiple(
        chunk_output_shape, patch_output_shape
    ).astype(np.int64)
    chunk_input_shape = (chunk_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = _get_patch_top_left_info(
        img_shape, patch_input_shape, patch_output_shape
    )
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape
    patch_info_list = np.stack(
        [
            np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
            np.stack([patch_output_tl_list, patch_output_br_list], axis=1),
        ],
        axis=1,
    )

    chunk_input_tl_list, _ = _get_patch_top_left_info(
        img_shape, chunk_input_shape, chunk_output_shape
    )
    chunk_input_br_list = chunk_input_tl_list + chunk_input_shape
    # * correct the coord so it stay within source image
    # * 校正坐标使其保持在源图像范围内
    y_sel = np.nonzero(chunk_input_br_list[:, 0] > img_shape[0])[0]
    x_sel = np.nonzero(chunk_input_br_list[:, 1] > img_shape[1])[0]
    chunk_input_br_list[y_sel, 0] = (
        img_shape[0] - patch_diff_shape[0]
    ) - chunk_input_tl_list[y_sel, 0]
    chunk_input_br_list[x_sel, 1] = (
        img_shape[1] - patch_diff_shape[1]
    ) - chunk_input_tl_list[x_sel, 1]
    chunk_input_br_list[y_sel, 0] = round_to_multiple(
        chunk_input_br_list[y_sel, 0], patch_output_shape[0]
    )
    chunk_input_br_list[x_sel, 1] = round_to_multiple(
        chunk_input_br_list[x_sel, 1], patch_output_shape[1]
    )
    chunk_input_br_list[y_sel, 0] += chunk_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    chunk_input_br_list[x_sel, 1] += chunk_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    chunk_output_tl_list = chunk_input_tl_list + patch_diff_shape // 2
    chunk_output_br_list = chunk_input_br_list - patch_diff_shape // 2  # may off pixels
    # 可能会有像素偏差
    chunk_info_list = np.stack(
        [
            np.stack([chunk_input_tl_list, chunk_input_br_list], axis=1),
            np.stack([chunk_output_tl_list, chunk_output_br_list], axis=1),
        ],
        axis=1,
    )

    return chunk_info_list, patch_info_list


####
def _post_proc_para_wrapper(pred_map_mmap_path, tile_info, func, func_kwargs):
    """Wrapper for parallel post processing."""
    """并行后处理的包装器。"""
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode="r")
    tile_pred_map = wsi_pred_map_ptr[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]]
    tile_pred_map = np.array(tile_pred_map)  # from mmap to ram
    # 从 mmap 到 ram
    return func(tile_pred_map, **func_kwargs), tile_info


####
def _assemble_and_flush(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    """Assemble the results. Write to newly created holder for this wsi"""
    """组装结果。写入为此 wsi 新创建的容器"""
    wsi_pred_map_ptr = np.load(wsi_pred_map_mmap_path, mmap_mode="r+")
    chunk_pred_map = wsi_pred_map_ptr[
        chunk_info[1][0][0] : chunk_info[1][1][0],
        chunk_info[1][0][1] : chunk_info[1][1][1],
    ]
    if patch_output_list is None:
        # chunk_pred_map[:] = 0 # zero flush when there is no-results
        # chunk_pred_map[:] = 0 # 没有结果时清零刷新
        # print(chunk_info.flatten(), 'flush 0')
        # 打印(chunk_info.flatten(), 'flush 0')
        return

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        chunk_pred_map[
            pcoord[0] : pcoord[0] + pdata.shape[0],
            pcoord[1] : pcoord[1] + pdata.shape[1],
        ] = pdata
    # print(chunk_info.flatten(), 'pass')
    # 打印(chunk_info.flatten(), 'pass')
    return


####
class InferManager(base.InferManager):
    def __run_model(self, patch_top_left_list, pbar_desc):
        # TODO: the cost of creating dataloader may not be cheap ?
        # TODO: 创建 dataloader 的成本可能不便宜？
        dataset = SerializeArray(
            "%s/cache_chunk.npy" % self.cache_path,
            patch_top_left_list,
            self.patch_input_shape,
        )

        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=self.batch_size,
            drop_last=False,
        )

        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=int(len(dataloader)),
            ncols=80,
            ascii=True,
            position=0,
        )

        # run inference on input patches
        # 对输入补丁运行推理
        accumulated_patch_output = []
        for batch_idx, batch_data in enumerate(dataloader):
            sample_data_list, sample_info_list = batch_data
            sample_output_list = self.run_step(sample_data_list)
            sample_info_list = sample_info_list.numpy()
            curr_batch_size = sample_output_list.shape[0]
            sample_output_list = np.split(sample_output_list, curr_batch_size, axis=0)
            sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
            sample_output_list = list(zip(sample_info_list, sample_output_list))
            accumulated_patch_output.extend(sample_output_list)
            pbar.update()
        pbar.close()
        return accumulated_patch_output

    def __select_valid_patches(self, patch_info_list, has_output_info=True):
        """Select valid patches from the list of input patch information.
        """从输入补丁信息列表中选择有效补丁。

        Args:
        参数:
            patch_info_list: patch input coordinate information
            patch_info_list: 补丁输入坐标信息
            has_output_info: whether output information is given
            has_output_info: 是否给出输出信息
        
        """
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        selected_indices = []
        for idx in range(patch_info_list.shape[0]):
            patch_info = patch_info_list[idx]
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            # 获取掩码对应放大倍数下的框
            if has_output_info:
                output_bbox = patch_info[1] * down_sample_ratio
            else:
                output_bbox = patch_info * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            # 补丁输出的坐标（即中心区域）
            output_roi = self.wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            if np.sum(output_roi) > 0:
                selected_indices.append(idx)
        sub_patch_info_list = patch_info_list[selected_indices]
        return sub_patch_info_list

    def __get_raw_prediction(self, chunk_info_list, patch_info_list):
        """Process input tiles (called chunks for inference) with HoVer-Net.
        """使用 HoVer-Net 处理输入切片（推理时称为块）。

        Args:
        参数:
            chunk_info_list: list of inference tile coordinate information
            chunk_info_list: 推理切片坐标信息列表
            patch_info_list: list of patch coordinate information
            patch_info_list: 补丁坐标信息列表
        
        """
        # 1 dedicated thread just to write results back to disk
        # 1 个专用线程仅用于将结果写回磁盘
        proc_pool = Pool(processes=1)
        wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path

        masking = lambda x, a, b: (a <= x) & (x <= b)
        for idx in range(0, chunk_info_list.shape[0]):
            chunk_info = chunk_info_list[idx]
            # select patch basing on top left coordinate of input
            # 根据输入的左上角坐标选择补丁
            start_coord = chunk_info[0, 0]
            end_coord = chunk_info[0, 1] - self.patch_input_shape
            selection = masking(
                patch_info_list[:, 0, 0, 0], start_coord[0], end_coord[0]
            ) & masking(patch_info_list[:, 0, 0, 1], start_coord[1], end_coord[1])
            chunk_patch_info_list = np.array(
                patch_info_list[selection]
            )  # * do we need copy ?
            # * 我们需要复制吗？

            # further select only the patches within the provided mask
            # 进一步仅选择提供的掩码内的补丁
            chunk_patch_info_list = self.__select_valid_patches(chunk_patch_info_list)

            # there no valid patches, so flush 0 and skip
            # 没有有效补丁，因此清零并跳过
            if chunk_patch_info_list.shape[0] == 0:
                proc_pool.apply_async(
                    _assemble_and_flush, args=(wsi_pred_map_mmap_path, chunk_info, None)
                )
                continue

            # shift the coordinare from wrt slide to wrt chunk
            # 将坐标从相对于切片转换为相对于块
            chunk_patch_info_list -= chunk_info[:, 0]
            chunk_data = self.wsi_handler.read_region(
                chunk_info[0][0][::-1], (chunk_info[0][1] - chunk_info[0][0])[::-1]
            )
            chunk_data = np.array(chunk_data)[..., :3]
            np.save("%s/cache_chunk.npy" % self.cache_path, chunk_data)

            pbar_desc = "Process Chunk %d/%d" % (idx, chunk_info_list.shape[0])
            patch_output_list = self.__run_model(
                chunk_patch_info_list[:, 0, 0], pbar_desc
            )

            proc_pool.apply_async(
                _assemble_and_flush,
                args=(wsi_pred_map_mmap_path, chunk_info, patch_output_list),
            )
        proc_pool.close()
        proc_pool.join()
        return

    def __dispatch_post_processing(self, tile_info_list, callback):
        """Post processing initialisation."""
        """后处理初始化。"""
        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        future_list = []
        wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]

            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                "nr_types": self.method["model_args"]["nr_types"],
                "return_centroids": True,
            }

            # TODO: standarize protocol
            # TODO: 标准化协议
            if proc_pool is not None:
                proc_future = proc_pool.submit(
                    _post_proc_para_wrapper,
                    wsi_pred_map_mmap_path,
                    tile_info,
                    self.post_proc_func,
                    func_kwargs,
                )
                # ! manually poll future and call callback later as there is no guarantee
                # ! 手动轮询 future 并稍后调用回调，因为不能保证
                # ! that the callback is called from main thread
                # ! 回调从主线程调用
                future_list.append(proc_future)
            else:
                results = _post_proc_para_wrapper(
                    wsi_pred_map_mmap_path, tile_info, self.post_proc_func, func_kwargs
                )
                callback(results)
        if proc_pool is not None:
            silent_crash = False
            # loop over all to check state a.k.a polling
            # 遍历所有以检查状态即轮询
            for future in as_completed(future_list):
                # ! silent crash, cancel all and raise error
                # ! 静默崩溃，取消所有并抛出错误
                if future.exception() is not None:
                    silent_crash = True
                    # ! cancel somehow leads to cascade error later
                    # ! 取消不知为何会导致后续的级联错误
                    # ! so just poll it then crash once all future
                    # ! 所以只需轮询它，然后在所有 future 都获取后崩溃一次
                    # ! acquired for now
                    # ! 现在已获取
                    # for future in future_list:
                    #     future.cancel()
                    # break
                else:
                    callback(future.result())
            assert not silent_crash
        return

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        """解析命令行参数并设置为实例变量。"""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        # 转换为元组
        self.chunk_shape = [self.chunk_shape, self.chunk_shape]
        self.tile_shape = [self.tile_shape, self.tile_shape]
        self.patch_input_shape = [self.patch_input_shape, self.patch_input_shape]
        self.patch_output_shape = [self.patch_output_shape, self.patch_output_shape]
        return

    def process_single_file(self, wsi_path, msk_path, output_dir):
        """Process a single whole-slide image and save the results.
        """处理单个全切片图像并保存结果。

        Args:
        参数:
            wsi_path: path to input whole-slide image
            wsi_path: 输入全切片图像的路径
            msk_path: path to input mask. If not supplied, mask will be automatically generated.
            msk_path: 输入掩码的路径。如果未提供，将自动生成掩码。
            output_dir: path where output will be saved
            output_dir: 输出将保存的路径

        """
        # TODO: customize universal file handler to sync the protocol
        # TODO: 定制通用文件处理器以同步协议
        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        chunk_input_shape = np.array(self.chunk_shape)
        patch_input_shape = np.array(self.patch_input_shape)
        patch_output_shape = np.array(self.patch_output_shape)

        path_obj = pathlib.Path(wsi_path)
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        start = time.perf_counter()
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext)
        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        self.wsi_handler.prepare_reading(
            read_mag=self.proc_mag, cache_path="%s/src_wsi.npy" % self.cache_path
        )
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1])  # to Y, X
        # 转换为 Y, X

        if msk_path is not None and os.path.isfile(msk_path):
            self.wsi_mask = cv2.imread(msk_path)
            self.wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
            self.wsi_mask[self.wsi_mask > 0] = 1
        else:
            log_info(
                "WARNING: No mask found, generating mask via thresholding at 1.25x!"
            )

            from skimage import morphology

            # simple method to extract tissue regions using intensity thresholding and morphological operations
            # 使用强度阈值处理和形态学操作提取组织区域的简单方法
            def simple_get_mask():
                scaled_wsi_mag = 1.25  # ! hard coded
                # ! 硬编码
                wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=scaled_wsi_mag)
                gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                mask = morphology.remove_small_objects(
                    mask == 0, min_size=16 * 16, connectivity=2
                )
                mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
                mask = morphology.binary_dilation(mask, morphology.disk(16))
                return mask

            self.wsi_mask = np.array(simple_get_mask() > 0, dtype=np.uint8)
        if np.sum(self.wsi_mask) == 0:
            log_info("Skip due to empty mask!")
            return
        if self.save_mask:
            cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), self.wsi_mask * 255)
        if self.save_thumb:
            wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
            cv2.imwrite(
                "%s/thumb/%s.png" % (output_dir, wsi_name),
                cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR),
            )

        # * declare holder for output
        # * 声明输出容器
        # create a memory-mapped .npy file with the predefined dimensions and dtype
        # 使用预定义的维度和 dtype 创建内存映射的 .npy 文件
        # TODO: dynamicalize this, retrieve from model?
        # TODO: 动态化这个，从模型中检索？
        out_ch = 3 if self.method["model_args"]["nr_types"] is None else 4
        self.wsi_inst_info = {}
        # TODO: option to use entire RAM if users have too much available, would be faster than mmap
        # TODO: 如果用户有太多可用内存，选项使用整个 RAM，会比 mmap 更快
        self.wsi_inst_map = np.lib.format.open_memmap(
            "%s/pred_inst.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape),
            dtype=np.int32,
        )
        # self.wsi_inst_map[:] = 0 # flush fill
        # self.wsi_inst_map[:] = 0 # 刷新填充

        # warning, the value within this is uninitialized
        # 警告，其中的值未初始化
        self.wsi_pred_map = np.lib.format.open_memmap(
            "%s/pred_map.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            dtype=np.float32,
        )
        # ! for debug
        # ! 用于调试
        # self.wsi_pred_map = np.load('%s/pred_map.npy' % self.cache_path, mmap_mode='r')
        end = time.perf_counter()
        log_info("Preparing Input Output Placement: {0}".format(end - start))

        # * raw prediction
        # * 原始预测
        start = time.perf_counter()
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
            self.wsi_proc_shape,
            chunk_input_shape,
            patch_input_shape,
            patch_output_shape,
        )

        # get the raw prediction of HoVer-Net, given info of inference tiles and patches
        # 给定推理切片和补丁的信息，获取 HoVer-Net 的原始预测
        self.__get_raw_prediction(chunk_info_list, patch_info_list)
        end = time.perf_counter()
        log_info("Inference Time: {0}".format(end - start))

        # TODO: deal with error banding
        # TODO: 处理误差带
        ##### * post processing
        ##### * 后处理
        ##### * done in 3 stages to ensure that nuclei at the boundaries are dealt with accordingly
        ##### * 分 3 个阶段完成，以确保边界处的细胞核得到相应处理
        start = time.perf_counter()
        tile_coord_set = _get_tile_info(self.wsi_proc_shape, tile_shape, ambiguous_size)
        # 3 sets of patches are extracted and are dealt with differently
        # 提取 3 组补丁，并以不同方式处理
        # tile_grid_info: central region of post processing tiles
        # tile_grid_info: 后处理切片的中心区域
        # tile_boundary_info: boundary region of post processing tiles
        # tile_boundary_info: 后处理切片的边界区域
        # tile_cross_info: region at corners of post processing tiles
        # tile_cross_info: 后处理切片角落的区域
        tile_grid_info, tile_boundary_info, tile_cross_info = tile_coord_set
        tile_grid_info = self.__select_valid_patches(tile_grid_info, False)
        tile_boundary_info = self.__select_valid_patches(tile_boundary_info, False)
        tile_cross_info = self.__select_valid_patches(tile_cross_info, False)

        ####################### * Callback can only receive 1 arg
        ####################### * 回调只能接收 1 个参数
        def post_proc_normal_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                # 外部
                return  # when there is nothing to do
                # 当无事可做时

            top_left = pos_args[1][::-1]

            # ! WARNING:
            # ! 警告:
            # ! inst ID may not be contiguous,
            # ! 实例 ID 可能不连续,
            # ! hence must use max as safeguard
            # ! 因此必须使用 max 作为保障

            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())
            for inst_id, inst_info in inst_info_dict.items():
                # now correct the coordinate wrt to wsi
                # 现在校正相对于 wsi 的坐标
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ] = pred_inst

            pbar.update()  # external
            # 外部
            return

        ####################### * Callback can only receive 1 arg
        ####################### * 回调只能接收 1 个参数
        def post_proc_fixing_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                # 外部
                return  # when there is nothing to do
                # 当无事可做时

            top_left = pos_args[1][::-1]

            # for fixing the boundary, keep all nuclei split at boundary (i.e within unambigous region)
            # 为了修复边界，保留所有在边界处分裂的细胞核（即在明确区域内）
            # of the existing prediction map, and replace all nuclei within the region with newly predicted
            # 在现有预测图中，并用新预测的替换该区域内的所有细胞核

            # ! WARNING:
            # ! 警告:
            # ! inst ID may not be contiguous,
            # ! 实例 ID 可能不连续,
            # ! hence must use max as safeguard
            # ! 因此必须使用 max 作为保障

            # ! must get before the removal happened
            # ! 必须在移除发生之前获取
            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())

            # * exclude ambiguous out from old prediction map
            # * 从旧预测图中排除模糊区域
            # check 1 pix of 4 edges to find nuclei split at boundary
            # 检查 4 条边的 1 个像素以找到在边界处分裂的细胞核
            roi_inst = self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ]
            roi_inst = np.copy(roi_inst)
            roi_edge = np.concatenate(
                [roi_inst[[0, -1], :].flatten(), roi_inst[:, [0, -1]].flatten()]
            )
            roi_boundary_inst_list = np.unique(roi_edge)[1:]  # exclude background
            # 排除背景
            roi_inner_inst_list = np.unique(roi_inst)[1:]
            roi_inner_inst_list = np.setdiff1d(
                roi_inner_inst_list, roi_boundary_inst_list, assume_unique=True
            )
            roi_inst = _remove_inst(roi_inst, roi_inner_inst_list)
            self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ] = roi_inst
            for inst_id in roi_inner_inst_list:
                self.wsi_inst_info.pop(inst_id, None)

            # * exclude unambiguous out from new prediction map
            # * 从新预测图中排除明确区域
            # check 1 pix of 4 edges to find nuclei split at boundary
            # 检查 4 条边的 1 个像素以找到在边界处分裂的细胞核
            roi_edge = pred_inst[roi_inst > 0]  # remove all overlap
            # 移除所有重叠
            boundary_inst_list = np.unique(roi_edge)  # no background to exclude
            # 没有背景需要排除
            inner_inst_list = np.unique(pred_inst)[1:]
            inner_inst_list = np.setdiff1d(
                inner_inst_list, boundary_inst_list, assume_unique=True
            )
            pred_inst = _remove_inst(pred_inst, boundary_inst_list)

            # * proceed to overwrite
            # * 继续覆盖
            for inst_id in inner_inst_list:
                # ! happen because we alrd skip thoses with wrong
                # ! 发生是因为我们已经跳过了那些错误的
                # ! contour (<3 points) within the postproc, so
                # ! 轮廓（<3 个点）在后处理中，所以
                # ! sanity gate here
                # ! 这里是完整性检查
                if inst_id not in inst_info_dict:
                    log_info("Nuclei id=%d not in saved dict WRN1." % inst_id)
                    continue
                inst_info = inst_info_dict[inst_id]
                # now correct the coordinate wrt to wsi
                # 现在校正相对于 wsi 的坐标
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            pred_inst = roi_inst + pred_inst
            self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ] = pred_inst

            pbar.update()  # external
            # 外部
            return

        #######################
        pbar_creator = lambda x, y: tqdm.tqdm(
            desc=y, leave=True, total=int(len(x)), ncols=80, ascii=True, position=0
        )
        pbar = pbar_creator(tile_grid_info, "Post Proc Phase 1")
        # * must be in sequential ordering
        # * 必须按顺序排列
        self.__dispatch_post_processing(tile_grid_info, post_proc_normal_tile_callback)
        pbar.close()

        pbar = pbar_creator(tile_boundary_info, "Post Proc Phase 2")
        self.__dispatch_post_processing(
            tile_boundary_info, post_proc_fixing_tile_callback
        )
        pbar.close()

        pbar = pbar_creator(tile_cross_info, "Post Proc Phase 3")
        self.__dispatch_post_processing(tile_cross_info, post_proc_fixing_tile_callback)
        pbar.close()

        end = time.perf_counter()
        log_info("Total Post Proc Time: {0}".format(end - start))

        # ! cant possibly save the inst map at high res, too large
        # ! 不可能以高分辨率保存实例图，太大了
        start = time.perf_counter()
        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)
        self.__save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
        end = time.perf_counter()
        log_info("Save Time: {0}".format(end - start))

    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.
        """处理全切片图像列表。

        Args:
        参数:
            run_args: arguments as defined in run_infer.py
            run_args: run_infer.py 中定义的参数
        
        """
        self._parse_args(run_args)

        if not os.path.exists(self.cache_path):
            rm_n_mkdir(self.cache_path)

        if not os.path.exists(self.output_dir + "/json/"):
            rm_n_mkdir(self.output_dir + "/json/")
        if self.save_thumb:
            if not os.path.exists(self.output_dir + "/thumb/"):
                rm_n_mkdir(self.output_dir + "/thumb/")
        if self.save_mask:
            if not os.path.exists(self.output_dir + "/mask/"):
                rm_n_mkdir(self.output_dir + "/mask/")

        wsi_path_list = glob.glob(self.input_dir + "/*")
        wsi_path_list.sort()  # ensure ordering
        # 确保顺序
        for wsi_path in wsi_path_list[:]:
            if os.path.isdir(wsi_path):
                continue
            wsi_base_name = pathlib.Path(wsi_path).stem
            msk_path = "%s/%s.png" % (self.input_mask_dir, wsi_base_name)
            if self.save_thumb or self.save_mask:
                output_file = "%s/json/%s.json" % (self.output_dir, wsi_base_name)
            else:
                output_file = "%s/%s.json" % (self.output_dir, wsi_base_name)
            if os.path.exists(output_file):
                log_info("Skip: %s" % wsi_base_name)
                continue
            try:
                log_info("Process: %s" % wsi_base_name)
                self.process_single_file(wsi_path, msk_path, self.output_dir)
                log_info("Finish")
            except Exception:
                logging.exception("Crash")
        rm_n_mkdir(self.cache_path)  # clean up all cache
        # 清理所有缓存
        return
