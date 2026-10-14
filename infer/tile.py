import logging
import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
# ! 必须放在顶部以便 VScode 调试
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import re
import sys
import warnings
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool

import cv2
import numpy as np
import psutil
import scipy.io as sio
import torch
import torch.utils.data as data
import tqdm
from dataloader.infer_loader import SerializeArray, SerializeFileList
from misc.utils import (
    color_deconvolution,
    cropping_center,
    get_bounding_box,
    log_debug,
    log_info,
    rm_n_mkdir,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage import color

import convert_format
from . import base


####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.
    
    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img
        
    """
    """为切片处理准备补丁信息。
    
    参数:
        img: 原始输入图像
        window_size: 输入补丁大小
        mask_size: 输出补丁大小
        return_src_top_corner: 是否返回图像左上角的坐标信息
        
    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # generating subpatches index from orginal
    # 从原始图像生成子补丁索引
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    if not return_src_top_corner:
        return img, patch_info
    else:
        return img, patch_info, [padt, padl]


####
def _post_process_patches(
    post_proc_func, post_proc_kwargs, patch_info, image_info, overlay_kwargs,
):
    """Apply post processing to patches.
    
    Args:
        post_proc_func: post processing function to use
        post_proc_kwargs: keyword arguments used in post processing function
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    """对补丁应用后处理。
    
    参数:
        post_proc_func: 要使用的后处理函数
        post_proc_kwargs: 后处理函数中使用的关键字参数
        patch_info: 补丁数据和相关信息
        image_info: 输入图像数据和相关信息
        overlay_kwargs: 叠加关键字参数

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    # 重新组装预测，根据补丁在原始图像中的位置排序
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)

    src_shape = image_info["src_shape"]
    src_image = image_info["src_image"]

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    # 裁剪回原始形状
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

    # * Implicit protocol
    # * 隐式协议
    # * a prediction map with instance of ID 1-N
    # * 具有 ID 1-N 实例的预测图
    # * and a dict contain the instance info, access via its ID
    # * 以及包含实例信息的字典，通过其 ID 访问
    # * each instance may have type
    # * 每个实例可能具有类型
    pred_inst, inst_info_dict = post_proc_func(pred_map, **post_proc_kwargs)

    overlaid_img = visualize_instances_dict(
        src_image.copy(), inst_info_dict, **overlay_kwargs
    )

    return image_info["name"], pred_map, pred_inst, inst_info_dict, overlaid_img


class InferManager(base.InferManager):
    """Run inference on tiles."""
    """在切片上运行推理。"""

    ####
    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        """
        处理单个图像切片，大小 < 5000x5000。
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        assert self.mem_usage < 1.0 and self.mem_usage > 0.0

        # * depend on the number of samples and their size, this may be less efficient
        # * 根据样本数量及其大小，这可能效率较低
        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        file_path_list = glob.glob(patterning("%s/*" % self.input_dir))
        file_path_list.sort()  # ensure same order
        # 确保相同的顺序
        assert len(file_path_list) > 0, 'Not Detected Any Files From Path'
        
        rm_n_mkdir(self.output_dir + '/json/')
        rm_n_mkdir(self.output_dir + '/mat/')
        rm_n_mkdir(self.output_dir + '/overlay/')
        if self.save_qupath:
            rm_n_mkdir(self.output_dir + "/qupath/")

        def proc_callback(results):
            """Post processing callback.
            
            Output format is implicit assumption, taken from `_post_process_patches`

            """
            """后处理回调。
            
            输出格式是隐式假设，取自 `_post_process_patches`

            """
            img_name, pred_map, pred_inst, inst_info_dict, overlaid_img = results

            nuc_val_list = list(inst_info_dict.values())
            # need singleton to make matlab happy
            # 需要单例以使 matlab 满意
            nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
            nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
            nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

            mat_dict = {
                "inst_map" : pred_inst,
                "inst_uid" : nuc_uid_list,
                "inst_type": nuc_type_list,
                "inst_centroid": nuc_coms_list
            }
            if self.nr_types is None: # matlab does not have None type array
                # matlab 没有 None 类型数组
                mat_dict.pop("inst_type", None) 

            if self.save_raw_map:
                mat_dict["raw_map"] = pred_map
            save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
            sio.savemat(save_path, mat_dict)

            save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))

            if self.save_qupath:
                nuc_val_list = list(inst_info_dict.values())
                nuc_type_list = np.array([v["type"] for v in nuc_val_list])
                nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])
                save_path = "%s/qupath/%s.tsv" % (self.output_dir, img_name)
                convert_format.to_qupath(
                    save_path, nuc_coms_list, nuc_type_list, self.type_info_dict
                )

            save_path = "%s/json/%s.json" % (self.output_dir, img_name)
            self.__save_json(save_path, inst_info_dict, None)
            return img_name

        def detach_items_of_uid(items_list, uid, nr_expected_items):
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True:
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            # 这样做是为了确保顺序
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        while len(file_path_list) > 0:

            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, "available")
            available_ram = int(available_ram * self.mem_usage)
            # available_ram >> 20 for MB, >> 30 for GB
            # available_ram >> 20 表示 MB，>> 30 表示 GB

            # TODO: this portion looks clunky but seems hard to detach into separate func
            # TODO: 这部分看起来很笨拙，但似乎很难分离到单独的函数中

            # * caching N-files into memory such that their expected (total) memory usage
            # * 将 N 个文件缓存到内存中，使其预期（总）内存使用量
            # * does not exceed the designated percentage of currently available memory
            # * 不超过当前可用内存的指定百分比
            # * the expected memory is a factor w.r.t original input file size and
            # * 预期内存是相对于原始输入文件大小的一个因子，并且
            # * must be manually provided
            # * 必须手动提供
            file_idx = 0
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)

                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                src_shape = img.shape

                img, patch_info, top_corner = _prepare_patching(
                    img, self.patch_input_shape, self.patch_output_shape, True
                )
                self_idx = np.full(patch_info.shape[0], file_idx, dtype=np.int32)
                patch_info = np.concatenate([patch_info, self_idx[:, None]], axis=-1)
                # ? may be expensive op
                # ? 可能是昂贵的操作
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                # * 这个 factor=5 仅适用于 HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0:
                    break

                file_idx += 1
                # if file_idx == 4: break
                # 如果 file_idx == 4: 中断
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                # TODO: refactor to explicit protocol
                # TODO: 重构为显式协议
                cache_image_info_list.append([src_shape, len(patch_info), top_corner])

            # * apply neural net on cached data
            # * 对缓存的数据应用神经网络
            dataset = SerializeFileList(
                cache_image_list, cache_patch_info_list, self.patch_input_shape
            )

            dataloader = data.DataLoader(
                dataset,
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
                drop_last=False,
            )

            pbar = tqdm.tqdm(
                desc="Process Patches",
                leave=True,
                total=int(len(cache_patch_info_list) / self.batch_size) + 1,
                ncols=80,
                ascii=True,
                position=0,
            )

            accumulated_patch_output = []
            for batch_idx, batch_data in enumerate(dataloader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self.run_step(sample_data_list)
                sample_info_list = sample_info_list.numpy()
                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(
                    sample_output_list, curr_batch_size, axis=0
                )
                sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
                sample_output_list = list(zip(sample_info_list, sample_output_list))
                accumulated_patch_output.extend(sample_output_list)
                pbar.update()
            pbar.close()

            # * parallely assemble the processed cache data for each file if possible
            # * 如果可能，并行组装每个文件的处理后缓存数据
            future_list = []
            for file_idx, file_path in enumerate(use_path_list):
                image_info = cache_image_info_list[file_idx]
                file_ouput_data, accumulated_patch_output = detach_items_of_uid(
                    accumulated_patch_output, file_idx, image_info[1]
                )

                # * detach this into func and multiproc dispatch it
                # * 将其分离到函数中并多进程调度
                src_pos = image_info[2]  # src top left corner within padded image
                # 填充图像内的源左上角
                src_image = cache_image_list[file_idx]
                src_image = src_image[
                    src_pos[0] : src_pos[0] + image_info[0][0],
                    src_pos[1] : src_pos[1] + image_info[0][1],
                ]

                base_name = pathlib.Path(file_path).stem
                file_info = {
                    "src_shape": image_info[0],
                    "src_image": src_image,
                    "name": base_name,
                }

                post_proc_kwargs = {
                    "nr_types": self.nr_types,
                    "return_centroids": True,
                }  # dynamicalize this
                # 动态化这个

                overlay_kwargs = {
                    "draw_dot": self.draw_dot,
                    "type_colour": self.type_info_dict,
                    "line_thickness": 2,
                }
                func_args = (
                    self.post_proc_func,
                    post_proc_kwargs,
                    file_ouput_data,
                    file_info,
                    overlay_kwargs,
                )

                # dispatch for parallel post-processing
                # 调度并行后处理
                if proc_pool is not None:
                    proc_future = proc_pool.submit(_post_process_patches, *func_args)
                    # ! manually poll future and call callback later as there is no guarantee
                    # ! 手动轮询 future 并稍后调用回调，因为不能保证
                    # ! that the callback is called from main thread
                    # ! 回调从主线程调用
                    future_list.append(proc_future)
                else:
                    proc_output = _post_process_patches(*func_args)
                    proc_callback(proc_output)

            if proc_pool is not None:
                # loop over all to check state a.k.a polling
                # 遍历所有以检查状态即轮询
                for future in as_completed(future_list):
                    # TODO: way to retrieve which file crashed ?
                    # TODO: 检索哪个文件崩溃的方法？
                    # ! silent crash, cancel all and raise error
                    # ! 静默崩溃，取消所有并抛出错误
                    if future.exception() is not None:
                        log_info("Silent Crash")
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
                        file_path = proc_callback(future.result())
                        log_info("Done Assembling %s" % file_path)
        return
