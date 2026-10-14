"""convert_chkpt_tf2pytorch.py.

Convert checkpoint from previous repository trained with tensorflow to pytorch.

"""
"""convert_chkpt_tf2pytorch.py.

将之前 TensorFlow 仓库训练的检查点转换为 PyTorch 格式。

"""

import pandas as pd
import numpy as np
import torch

mapping = pd.read_csv("variables_tf2pytorch.csv", index_col=False)
mapping = {v["Tensorflow"]: v['Pytorch'] for k, v in mapping.T.to_dict().items()}

# mapping
# 映射表
tf_path = "" # to original tensorflow chkpt ends with .npz
# 原始 TensorFlow 检查点路径，以 .npz 结尾
pt_path = "" # to convert pytorch chkpt ends with .tar
# 转换后的 PyTorch 检查点保存路径，以 .tar 结尾

pt = {}
tf = np.load(tf_path)

for tf_k, tf_v in tf.items():
    if 'linear' in tf_k: continue # should only be for pretrained model 
    # 仅用于预训练模型
    pt_k = mapping[tf_k]
    if "conv" in pt_k and "bn" not in pt_k and "bias" not in pt_k:
        tf_v = np.transpose(tf_v, [3, 2, 0, 1])
    if "shortcut" in pt_k:
        tf_v = np.transpose(tf_v, [3, 2, 0, 1])
    pt[pt_k] = torch.from_numpy(tf_v)
# make compatible with repo structure
# 使其与仓库结构兼容
pt["upsample2x.unpool_mat"] = torch.from_numpy(np.ones((2, 2), dtype="float32"))
pt = {"desc": pt}
torch.save(pt, pt_path)
