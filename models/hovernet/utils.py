import math
import numpy as np

import torch
import torch.nn.functional as F

from matplotlib import cm


####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    """中心裁剪图像。

    参数:
        x: 输入图像
        cropping: 裁剪量
        data_format: 选择 `NCHW` 或 `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    """中心裁剪 x 使其形状与 y 相同。y 的维度必须小于 x 的维度。

    参数:
        x: 输入数组
        y: 具有目标形状的数组。

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    """交叉熵损失。假设为 NHWC 格式！

    参数:
        pred: 预测数组
        true: 真实标注数组
    
    返回:
        交叉熵损失

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    # 缩放预测值，使每个样本的类别概率和为 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    # 手动计算交叉熵
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    """`pred` 和 `true` 必须是 torch.float32 类型。假设形状为 NxHxWxC。"""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error

    """
    """计算均方误差损失。

    参数:
        true: 水平和垂直组合图的真实标注
        pred: 水平和垂直组合图的预测值
    
    返回:
        loss: 均方误差

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients

    """
    """计算水平和垂直图预测值梯度的均方误差。
    假设通道 0 是垂直方向，通道 1 是水平方向。

    参数:
        true:  水平和垂直组合图的真实标注
        pred:  水平和垂直组合图的预测值
        focus: 应用损失的区域（我们只计算细胞核内的损失）
    
    返回:
        loss:  梯度的均方误差

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        """获取指定大小的 Sobel 卷积核。"""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        """用于计算梯度。"""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        # 常量
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant
        # 常量

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW
        # Nx1xHxW

        # can only apply in NCHW mode
        # 只能在 NCHW 模式下应用
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        # 转换为 NHWC 格式
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    # 假设输入为 NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    # 使用聚焦区域进行人工 reduce_mean
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss
