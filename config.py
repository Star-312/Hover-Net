import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""
    """配置文件。"""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        # 开启调试标志，以便更容易追踪一些并行处理问题
        self.debug = False

        model_name = "hovernet"
        model_mode = "original" # choose either `original` or `fast`
        # 选择 `original`（原始模式）或 `fast`（快速模式）

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        nr_type = 5 # number of nuclear types (including background)
        # 细胞核类型数量（包括背景）

        # whether to predict the nuclear type, availability depending on dataset!
        # 是否预测细胞核类型，是否可用取决于数据集！
        self.type_classification = True

        # shape information - 
        # 形状信息 -
        # below config is for original mode. 
        # 以下配置适用于原始模式。
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # 如果使用原始模型模式，act_shape 和 out_shape 分别使用 [270,270] 和 [80,80]
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        # 如果使用快速模型模式，act_shape 和 out_shape 分别使用 [256,256] 和 [164,164]
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        # 数据增强时使用的图像块形状（更大的块可能减少边缘伪影）
        act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
        # 作为网络输入的图像块形状 - 在数据增强后进行中心裁剪
        out_shape = [80, 80] # patch shape at output of network
        # 网络输出的图像块形状

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "consep" # extracts dataset info from dataset.py
        # 从 dataset.py 中提取数据集信息
        self.log_dir = "logs/" # where checkpoints will be saved
        # 模型检查点保存路径

        # paths to training and validation patches
        # 训练和验证图像块的路径
        self.train_dir_list = [
            "train_patches_path"
        ]
        self.valid_dir_list = [
            "valid_patches_path"
        ]

        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        # * 将配置解析到运行状态，并设置相关变量
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
