"""公共工具包：数据、拼接、设备、ONNX 等。

导入本包任意子模块时，会自动设置 HF 镜像源，避免国内/代理环境访问
huggingface.co 被拦截返回 502。训练、推理、导出均依赖本包，因此都受益。
用户可用自己的 HF_ENDPOINT 覆盖。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
