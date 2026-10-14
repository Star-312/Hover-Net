"""run_train.py

Main HoVer-Net training script.

Usage:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

Options:
  -h --help       Show this string.
  --version       Show version.
  --gpu=<id>      Comma separated GPU list. [default: 0,1,2,3]
  --view=<dset>   Visualise images after augmentation. Choose 'train' or 'valid'.
"""
"""run_train.py

HoVer-Net 主训练脚本。

用法:
  run_train.py [--gpu=<id>] [--view=<dset>]
  run_train.py (-h | --help)
  run_train.py --version

选项:
  -h --help       显示此帮助信息。
  --version       显示版本号。
  --gpu=<id>      逗号分隔的 GPU 列表。[默认: 0,1,2,3]
  --view=<dset>   可视化数据增强后的图像。选择 'train' 或 'valid'。
"""

import cv2

cv2.setNumThreads(0)
import argparse
import glob
import importlib
import inspect
import json
import os
import shutil

import matplotlib
import numpy as np
import torch
from docopt import docopt
from tensorboardX import SummaryWriter
from torch.nn import DataParallel  # TODO: switch to DistributedDataParallel
# TODO: 切换为 DistributedDataParallel
from torch.utils.data import DataLoader

from config import Config
from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine
from run_utils.utils import (
    check_log_dir,
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
)


#### have to move outside because of spawn
# 必须移到外面，因为使用了 spawn（多进程启动方式）
# * must initialize augmentor per worker, else duplicated rng generators may happen
# * 必须为每个 worker 初始化增强器，否则可能会出现重复的随机数生成器
def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # ! 为了使种子链可复现，必须使用 torch 的随机数，而不是 numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # 主线程的 torch 随机数生成器会重新生成一个基础种子，然后
    # copied into the dataloader each time it created (i.e start of each epoch)
    # 在每次创建 dataloader 时复制进去（即每个 epoch 开始时）
    # then dataloader with this seed will spawn worker, now we reseed the worker
    # 然后带有此种子的 dataloader 会 spawn worker，现在我们为 worker 重新设置种子
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    # 为了使其更随机，只需将 torch.randint 替换为 np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # 打印 'Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed)
    # retrieve the dataset copied into this worker process
    # 检索复制到此 worker 进程中的数据集
    # then set the random seed for each augmentation
    # 然后为每次数据增强设置随机种子
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


####
class TrainManager(Config):
    """Either used to view the dataset or to initialise the main training loop."""
    """用于查看数据集或初始化主训练循环。"""

    def __init__(self):
        super().__init__()
        return

    ####
    def view_dataset(self, mode="train"):
        """
        Manually change to plt.savefig or plt.show 
        if using on headless machine or not
        """
        """
        手动切换为 plt.savefig 或 plt.show
        如果在无图形界面的机器上使用，请相应调整
        """
        self.nr_gpus = 1
        import matplotlib.pyplot as plt
        check_manual_seed(self.seed)
        # TODO: what if each phase want diff annotation ?
        # TODO: 如果每个阶段需要不同的标注怎么办？
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for batch_data in dataloader:  
            # convert from Tensor to Numpy
            # 将 Tensor 转换为 Numpy
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.show()
        self.nr_gpus = -1
        return

    ####
    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0

        # ! Hard assumption on file type
        # ! 对文件类型有硬性假设
        file_list = []
        if run_mode == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # to always ensure same input ordering
        # 始终确保相同的输入顺序

        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (run_mode, "%s_dir_list" % run_mode)
        )
        print("Dataset %s: %d" % (run_mode, len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode]
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    ####
    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        """Simply run the defined run_step of the related method once."""
        """简单地运行一次相关方法定义的 run_step。"""
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            # check_log_dir(log_dir)
            # 检查日志目录
            rm_n_mkdir(log_dir)

            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + "/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)  # create empty file
                # 创建空文件
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        ####
        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )
        ####
        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = "%s/%s_epoch=%d.tar" % (
                prev_phase_dir,
                net_name,
                max(epoch_list),
            )
            return last_chkpts_path

        # TODO: adding way to load pretrained weight or resume the training
        # TODO: 添加加载预训练权重或恢复训练的方式
        # parsing the network and optimizer information
        # 解析网络和优化器信息
        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(
                net_info["desc"]
            ), "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            # TODO: customize print-out for each run ?
            # TODO: 为每次运行定制打印输出？
            # summary_string(net_desc, (3, 270, 270), device='cpu')
            # summary_string(net_desc, (3, 270, 270), device='cpu')

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    # * depend on logging format so may be broken if logging format has been changed
                    # * 依赖日志格式，如果日志格式被更改可能会失效
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {
                            k: torch.from_numpy(v) for k, v in net_state_dict.items()
                        }
                    elif chkpt_ext == "tar":  # ! assume same saving format we desire
                        # ! 假设使用我们期望的相同保存格式
                        net_state_dict = torch.load(pretrained_path)["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(
                    "Model `%s` pretrained path: %s" % (colored_word, pretrained_path)
                )

                # load_state_dict returns (missing keys, unexpected keys)
                # load_state_dict 返回（缺失的键，意外的键）
                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                # * uncomment for your convenience
                # * 为方便起见取消注释
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            # * extremely slow to pass this on DGX with 1 GPU, why (?)
            # * 在单 GPU 的 DGX 上传递这个非常慢，为什么(?)
            net_desc = DataParallel(net_desc)
            net_desc = net_desc.to("cuda")
            # print(net_desc) # * dump network definition or not?
            # 打印(net_desc) # * 是否导出网络定义？
            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            # TODO: expand for external aug for scheduler
            # TODO: 为调度器扩展外部数据增强
            nr_iter = opt["nr_epochs"] * len(loader_dict["train"])
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # TODO: standardize API for external hooks
                # TODO: 标准化外部钩子的 API
                "extra_info": net_info["extra_info"],
            }

        # parsing the running engine configuration
        # 解析运行引擎配置
        assert (
            "train" in run_engine_opt
        ), "No engine for training detected in description file"

        # initialize runner and attach callback afterward
        # 初始化运行器并在之后挂载回调
        # * all engine shared the same network info declaration
        # * 所有引擎共享相同的网络信息声明
        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list, in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        # retrieve main runner
        # 获取主运行器
        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        # start the run loop
        # 开始运行循环
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")
        return

    ####
    def run(self):
        """Define multi-stage run or cross-validation or whatever in here."""
        """在这里定义多阶段运行或交叉验证或其他任何操作。"""
        self.nr_gpus = torch.cuda.device_count()
        print('Detect #GPUS: %d' % self.nr_gpus)

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            if len(phase_list) == 1:
                save_path = self.log_dir
            else:
                save_path = self.log_dir + "/%02d/" % (phase_idx)
            self.run_once(
                phase_info, engine_opt, save_path, prev_log_dir=prev_save_path
            )
            prev_save_path = save_path


####
if __name__ == "__main__":
    args = docopt(__doc__, version="HoVer-Net v1.0")
    trainer = TrainManager()

    if args["--view"]:
        if args["--view"] != "train" and args["--view"] != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args["--view"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
        trainer.run()
