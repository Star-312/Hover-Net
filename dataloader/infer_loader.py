import sys
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data

import psutil


####
class SerializeFileList(data.IterableDataset):
    """Read a single file as multiple patches of same shape, perform the padding beforehand."""
    """将单个文件读取为多个相同形状的图像块，预先进行填充。"""

    def __init__(self, img_list, patch_info_list, patch_size, preproc=None):
        super().__init__()
        self.patch_size = patch_size

        self.img_list = img_list
        self.patch_info_list = patch_info_list

        self.worker_start_img_idx = 0
        # * for internal worker state
        # * 用于内部 worker 状态
        self.curr_img_idx = 0
        self.stop_img_idx = 0
        self.curr_patch_idx = 0
        self.stop_patch_idx = 0
        self.preproc = preproc
        return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            # 单进程数据加载，返回完整迭代器
            self.stop_img_idx = len(self.img_list)
            self.stop_patch_idx = len(self.patch_info_list)
            return self
        else:  # in a worker process so split workload, return a reduced copy of self
            # 在 worker 进程中，因此拆分工作负载，返回 self 的精简副本
            per_worker = len(self.patch_info_list) / float(worker_info.num_workers)
            per_worker = int(math.ceil(per_worker))

            global_curr_patch_idx = worker_info.id * per_worker
            global_stop_patch_idx = global_curr_patch_idx + per_worker
            self.patch_info_list = self.patch_info_list[
                global_curr_patch_idx:global_stop_patch_idx
            ]
            self.curr_patch_idx = 0
            self.stop_patch_idx = len(self.patch_info_list)
            # * check img indexer, implicit protocol in infer.py
            # * 检查图像索引器，infer.py 中的隐式协议
            global_curr_img_idx = self.patch_info_list[0][-1]
            global_stop_img_idx = self.patch_info_list[-1][-1] + 1
            self.worker_start_img_idx = global_curr_img_idx
            self.img_list = self.img_list[global_curr_img_idx:global_stop_img_idx]
            self.curr_img_idx = 0
            self.stop_img_idx = len(self.img_list)
            return self  # does it mutate source copy?
            # 这会修改源副本吗？

    def __next__(self):

        if self.curr_patch_idx >= self.stop_patch_idx:
            raise StopIteration  # when there is nothing more to yield
            # 当没有更多内容可生成时
        patch_info = self.patch_info_list[self.curr_patch_idx]
        img_ptr = self.img_list[patch_info[-1] - self.worker_start_img_idx]
        patch_data = img_ptr[
            patch_info[0] : patch_info[0] + self.patch_size,
            patch_info[1] : patch_info[1] + self.patch_size,
        ]
        self.curr_patch_idx += 1
        if self.preproc is not None:
            patch_data = self.preproc(patch_data)
        return patch_data, patch_info


####
class SerializeArray(data.Dataset):
    def __init__(self, mmap_array_path, patch_info_list, patch_size, preproc=None):
        super().__init__()
        self.patch_size = patch_size

        # use mmap as intermediate sharing, else variable will be duplicated
        # 使用 mmap 作为中间共享，否则变量会被复制
        # accross torch worker => OOM error, open in read only mode
        # 跨 torch worker 会导致 OOM 错误，以只读模式打开
        self.image = np.load(mmap_array_path, mmap_mode="r")

        self.patch_info_list = patch_info_list
        self.preproc = preproc
        return

    def __len__(self):
        return len(self.patch_info_list)

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        patch_data = self.image[
            patch_info[0] : patch_info[0] + self.patch_size[0],
            patch_info[1] : patch_info[1] + self.patch_size[1],
        ]
        if self.preproc is not None:
            patch_data = self.preproc(patch_data)
        return patch_data, patch_info
