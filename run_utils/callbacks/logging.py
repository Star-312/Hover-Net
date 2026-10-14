import json
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from termcolor import colored

from .base import BaseCallbacks
from .serialize import fig2data, serialize

# TODO: logging for all printed info on the terminal
# TODO: 终端上所有打印信息的日志


####
class LoggingGradient(BaseCallbacks):
    """Will log per each training step."""
    """将记录每个训练步骤。"""

    def _pyplot_grad_flow(self, named_parameters):
        """Plots the gradients flowing through different layers in the net during training.
        "_pyplot_grad_flow(self.model.named_parameters())" to visualize the gradient flow.

        ! Very slow if triggered per steps because of CPU <=> GPU.

        """
        """绘制训练期间流经网络不同层的梯度。
        "_pyplot_grad_flow(self.model.named_parameters())" 用于可视化梯度流。

        ! 如果每步触发会非常慢，因为 CPU <=> GPU。

        """
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        fig = plt.figure(figsize=(10, 10))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # zoom in on the lower gradient regions
        # 放大低梯度区域
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend(
            [
                Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4),
            ],
            ["max-gradient", "mean-gradient", "zero-gradient"],
        )
        fig = np.transpose(fig2data(fig), axes=[2, 0, 1])  # HWC => CHW
        # HWC => CHW
        plt.close()
        return fig

    def run(self, state, event):

        if random.random() > 0.05:
            return
        curr_step = state.curr_global_step

        # logging the grad of all trainable parameters
        # 记录所有可训练参数的梯度
        tfwriter = state.log_info["tfwriter"]
        run_info = state.run_info
        for net_name, net_info in run_info.items():
            netdesc = net_info["desc"].module
            for param_name, param in netdesc.named_parameters():
                param_grad = param.grad
                # TODO: sync test None or epislon for pytorch 1.4 vs 1.5
                # TODO: 同步测试 pytorch 1.4 vs 1.5 的 None 或 epsilon
                if param_grad is None:
                    continue
                tfwriter.add_histogram(
                    "%s_grad/%s" % (net_name, param_name),
                    param_grad.detach().cpu().numpy().flatten(),
                    global_step=curr_step,
                )  # ditribute into 10 bins (np default)
                # 分布到 10 个 bin（numpy 默认）
                tfwriter.add_histogram(
                    "%s_para/%s" % (net_name, param_name),
                    param.detach().cpu().numpy().flatten(),
                    global_step=curr_step,
                )  # ditribute into 10 bins (np default)
                # 分布到 10 个 bin（numpy 默认）
        return


####
class LoggingEpochOutput(BaseCallbacks):
    """Must declare save dir first in the shared global state of the attached engine."""
    """必须首先在附加引擎的共享全局状态中声明保存目录。"""

    def __init__(self, per_n_epoch=1):
        super().__init__()
        self.per_n_epoch = per_n_epoch

    def run(self, state, event):

        # only logging every n epochs also
        # 也只每 n 个 epoch 记录一次
        if state.curr_epoch % self.per_n_epoch != 0:
            return

        # TODO: rename to differentiate global vs local epoch
        # TODO: 重命名以区分全局和局部 epoch
        if state.global_state is not None:
            current_epoch = str(state.global_state.curr_epoch)
        else:
            current_epoch = str(state.curr_epoch)

        output = state.tracked_step_output

        def get_serializable_values(output_format):
            log_dict = {}
            # get type and variable that is serializable
            # 获取可序列化的类型和变量
            # to console or other logging format (json, tensorboard)
            # 到控制台或其他日志格式（json、tensorboard）
            for variable_type, variable_dict in output.items():
                for value_name, value in variable_dict.items():
                    value_name = "%s-%s" % (state.attached_engine_name, value_name)
                    new_format = serialize(value, variable_type, output_format)
                    if new_format is not None:
                        log_dict[value_name] = new_format
            return log_dict

        # * Serialize to Console
        # * 序列化到控制台
        # align the console print output
        # 对齐控制台打印输出
        formatted_values = get_serializable_values("console")
        max_length = len(max(formatted_values.keys(), key=len))
        for value_name, value_text in formatted_values.items():
            value_name = colored(value_name.ljust(max_length), "green")
            print("------%s : %s" % (value_name, value_text))

        # TODO: [CRITICAL] fix passing this between engine
        # TODO: [关键] 修复在引擎之间传递这个
        # if not state.logging: return
        # if not state.logging: return

        # * Serialize to JSON file
        # * 序列化到 JSON 文件
        stat_dict = get_serializable_values("json")
        # json stat log file, update and overwrite
        # json 统计日志文件，更新并覆盖
        with open(state.log_info["json_file"]) as json_file:
            json_data = json.load(json_file)

        if current_epoch in json_data:
            old_stat_dict = json_data[current_epoch]
            stat_dict.update(old_stat_dict)
        current_epoch_dict = {current_epoch: stat_dict}
        json_data.update(current_epoch_dict)

        # TODO: may corrupt
        # TODO: 可能会损坏
        with open(state.log_info["json_file"], "w") as json_file:
            json.dump(json_data, json_file)

        # * Serialize to Tensorboard
        # * 序列化到 Tensorboard
        tfwriter = state.log_info["tfwriter"]
        formatted_values = get_serializable_values("tensorboard")
        # ! may need to flush to force update
        # ! 可能需要 flush 来强制更新
        for value_name, value in formatted_values.items():
            # TODO: dynamically call this
            # TODO: 动态调用这个
            if value[0] == "scalar":
                tfwriter.add_scalar(value_name, value[1], current_epoch)
            elif value[0] == "image":
                tfwriter.add_image(
                    value_name, value[1], current_epoch, dataformats="HWC"
                )
        # tfwriter.flush()
        # tfwriter.flush()

        return
