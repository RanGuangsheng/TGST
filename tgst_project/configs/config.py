"""
TGST 配置文件

实验基于 CWRU 轴承故障数据集 (Case Western Reserve University)
- 采样频率: 12kHz, DE端加速度计, 0HP载荷
- 信号长度: 512 点
- 初始4类 + 1扩展类(Inner_0.014")
"""
import os

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
CWRU_DIR = os.path.join(DATASETS_DIR, "CWRU")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ===================== 模型配置 =====================
MODEL_CONFIG = {
    "signal_dim": 1,           # 信号维度 (单通道振动)
    "d_model": 64,             # 模型隐藏维度 (受6GB VRAM限制)
    "n_heads": 4,              # 注意力头数
    "n_encoder_layers": 2,     # Transformer层数
    "d_ff": 256,               # FFN中间维度
    "dropout": 0.1,
    "adapter_dim": 64,          # 适配器瓶颈维度
    "bert_model": "bert-base-chinese",
}

# ===================== 数据配置 =====================
DATA_CONFIG = {
    "fs": 12000,                # CWRU 采样频率 (12kHz, DE端)
    "signal_length": 512,       # 信号片段长度
    "n_samples_per_class": 100, # 每类样本数
    "n_classes": 4,             # 初始类别数
    "train_ratio": 0.7,
    "batch_size": 8,
}

# ===================== 训练配置 =====================
TRAIN_CONFIG = {
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "lambda_cl": 1.0,           # 对比损失权重 (1.0 激活语义指导)
    "temperature": 0.01,       # 对比损失温度 (0.01 增强区分度)
    "max_grad_norm": 1.0,
}

# ===================== 扩展配置 =====================
EXTENSION_CONFIG = {
    "ext_samples": 100,        # 扩展每类样本数
    "ext_epochs": 20,
    "ext_lr": 5e-4,
}

# ===================== 故障类别定义 =====================
# 类别名称仅用于显示，不输入模型
# 文本描述由 text_generator_v3.py 生成，描述信号特征而非故障类型
FAULT_CLASSES = {
    0: {"name": "正常",       "cwru_files": [97, 98, 99]},
    1: {"name": "内圈0.007",  "cwru_files": [107, 108]},
    2: {"name": "外圈0.007",  "cwru_files": [135, 136, 137]},
    3: {"name": "滚珠0.007",  "cwru_files": [159, 160, 161]},
    4: {"name": "内圈0.014",  "cwru_files": [120, 121]},  # 扩展类
}

# ===================== 设备配置 =====================
DEVICE = "cuda"  # 会自动检测


def get_config():
    return {
        "model": MODEL_CONFIG,
        "data": DATA_CONFIG,
        "train": TRAIN_CONFIG,
        "extension": EXTENSION_CONFIG,
        "fault_classes": FAULT_CLASSES,
        "device": DEVICE,
    }
