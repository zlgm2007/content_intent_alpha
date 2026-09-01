"""训练设备选择工具。

按优先级自动选择最佳设备：
    CUDA（NVIDIA GPU） > MPS（Apple Silicon M 系列） > CPU（通用兜底）

这样在 MacBook M 系列上自动走 MPS 加速，在非苹果 / 非 M 系列的服务器上
（无 CUDA 时）自动回退到 CPU，保证各类型机器都能正常训练。
"""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """返回最佳可用训练设备，并打印选择结果。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] 使用 CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] 使用 MPS (Apple Silicon GPU 加速)")
    else:
        device = torch.device("cpu")
        print("[device] 使用 CPU (未检测到 CUDA / MPS，兼容模式)")
    return device
