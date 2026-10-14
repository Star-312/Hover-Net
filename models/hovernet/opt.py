import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from .targets import gen_targets, prep_sample
from .net_desc import create_model
from .run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output


# TODO: training config only ?
# TODO: 仅训练配置？
# TODO: switch all to function name String for all option
# TODO: 所有选项都切换为函数名字符串
def get_config(nr_type, mode):
    return {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # ! 所有阶段都有相同数量的运行引擎
        # phases are run sequentially from index 0 to N
        # 阶段从索引 0 到 N 顺序运行
        "phase_list": [
            {
                "run_info": {
                    # may need more dynamic for each network
                    # 每个网络可能需要更多动态配置
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, nr_types=nr_type, 
                            freeze=True, mode=mode
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                # 应与优化器内参数的关键字匹配
                                "lr": 1.0e-4,  # initial learning rate,
                                # 初始学习率，
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        # 学习率调度器
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # 加载路径，-1 表示自动加载上一阶段的检查点，
                        # None to start from scratch
                        # None 表示从头开始
                        "pretrained": "../pretrained/ImageNet-ResNet50-Preact_pytorch.tar",
                        # 'pretrained': None,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 16, "valid": 16,},  # engine name : value
                # 引擎名称 : 值
                "nr_epochs": 50,
            },
            {
                "run_info": {
                    # may need more dynamic for each network
                    # 每个网络可能需要更多动态配置
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, nr_types=nr_type, 
                            freeze=False, mode=mode
                        ),
                        "optimizer": [
                            optim.Adam,
                            {  # should match keyword for parameters within the optimizer
                                # 应与优化器内参数的关键字匹配
                                "lr": 1.0e-4,  # initial learning rate,
                                # 初始学习率，
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        # 学习率调度器
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # 加载路径，-1 表示自动加载上一阶段的检查点，
                        # None to start from scratch
                        # None 表示从头开始
                        "pretrained": -1,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 4, "valid": 8,}, # batch size per gpu
                # 每个 GPU 的批量大小
                "nr_epochs": 50,
            },
        ],
        # ------------------------------------------------------------------
        # TODO: dynamically for dataset plugin selection and processing also?
        # TODO: 数据集插件选择和处理也动态化？
        # all enclosed engine shares the same neural networks
        # 所有包含的引擎共享相同的神经网络
        # as the on at the outer calling it
        # 与外部调用它的引擎相同
        "run_engine": {
            "train": {
                # TODO: align here, file path or what? what about CV?
                # TODO: 在这里对齐，文件路径还是什么？CV 怎么办？
                "dataset": "",  # whats about compound dataset ?
                # 复合数据集怎么办？
                "nr_procs": 16,  # number of threads for dataloader
                # dataloader 的线程数
                "run_step": train_step,  # TODO: function name or function variable ?
                # TODO: 函数名还是函数变量？
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                # 回调按事件的列表顺序运行
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        # LoggingGradient(), # TODO: 非常慢，可能是由于 tensor/numpy 来回转换？
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "",  # whats about compound dataset ?
                # 复合数据集怎么办？
                "nr_procs": 8,  # number of threads for dataloader
                # dataloader 的线程数
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # * 停止聚合上一次运行的输出等
                # callbacks are run according to the list order of the event
                # 回调按事件的列表顺序运行
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        # TODO: 有办法预加载这些吗？
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, nr_types=nr_type)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }
