"""
CWRU Dataset V2 — 信号特征描述型文本标签

核心设计原则:
1. 文本描述信号特征（频域/时域/统计/调制），不描述故障类型
2. 不出现"内圈""外圈""滚珠""正常"等故障类型词
3. 每类 8 条变体，训练时随机分配，避免文本->标签的死记映射
4. 类间文本描述的信号特征有语义区分度
5. 类内变体用词不同但语义相关，BERT 需理解语义才能提取指导

初始 4 类:
  0: Normal (97, 98, 99)        — 平稳无冲击
  1: Inner 0.007" (107, 108)   — 周期冲击+转频调制(中等)
  2: Outer 0.007" (135,136,137)— 等间隔冲击+无调制(中等)
  3: Ball 0.007" (159,160,161) — 非等间隔冲击+低频调制

扩展类:
  4: Inner 0.014" (120, 121)   — 强周期冲击+深调制(高强度)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from functools import partial
import random

# ── CWRU file mapping ──────────────────────────────────────────
CWRU_FILES = {
    'Normal':      [97, 98, 99],
    'Inner_0.007': [107, 108],
    'Outer_0.007': [135, 136, 137],
    'Ball_0.007':  [159, 160, 161],
    'Inner_0.014': [120, 121],
}

INITIAL_CLASSES = [0, 1, 2, 3]
EXTENSION_CLASSES = [4]

# ── 信号特征描述型文本标签 (每类 8 条变体) ──────────────────────
# 描述维度: 频域特征 / 时域特征 / 统计特征 / 调制特征 / 频带能量
# ⚠️ 不含故障类型词，只描述信号现象
TEXT_VARIANTS = {
    # 类0: 平稳无冲击 — 低能量、高斯分布、无瞬态、频谱平坦
    0: [
        "振动幅值较小 波形平稳无显著冲击 各频段能量均匀分布",
        "时域信号呈高斯分布 峭度接近三 无周期性脉冲成分",
        "频谱平坦无明显共振峰 能量集中于低频段 无瞬态冲击",
        "信号平稳随机振动 均方根值低 无调制现象 无冲击响应",
        "各频带能量分布均衡 无主导频率分量 波形无突变",
        "时域波形无明显波峰 信号变化平缓 统计特征稳定",
        "宽带频谱能量分散 无单一频带主导 冲击能量为零",
        "信号整体能量低 无周期性成分 幅值分布对称集中",
    ],
    # 类1: 周期冲击+转频调制(中等) — 等间隔脉冲、幅值受转频调制、高频共振
    1: [
        "高频段出现周期性冲击 冲击间隔与转轴周期相关 存在幅值调制",
        "共振频带能量显著 时域波形呈现等间隔脉冲 调制频率与转速一致",
        "峭度值偏高 信号呈现非高斯分布 存在周期性瞬态成分",
        "高频共振区域能量集中 冲击幅值随转频周期变化 包络谱存在转频边带",
        "时域出现周期性冲击序列 间隔规律 高频分量丰富 低频能量较低",
        "信号呈现周期性调制特征 冲击重复频率与转频相关 包络谱谐波明显",
        "频谱高频段存在共振带 冲击成分周期性出现 调制深度中等",
        "瞬态冲击以等间隔出现 幅值受转频调制 高频共振响应中等",
    ],
    # 类2: 等间隔冲击+无调制(中等) — 固定间隔、幅值稳定、无转频调制
    2: [
        "高频段出现等间隔冲击 冲击间隔固定不受转速调制 能量集中",
        "时域波形呈现稳定周期脉冲 冲击幅值较为均匀 无转频调制",
        "共振频带响应明显 冲击间隔恒定不随时间变化 包络谱峰值单一",
        "信号存在周期性冲击成分 间隔固定 峭度较高 无幅值调制现象",
        "高频区域呈现规律性能量分布 冲击重复频率稳定 调制效应弱",
        "频谱共振带能量集中 冲击以恒定间隔出现 无边带调制",
        "时域脉冲间隔恒定 幅值波动小 无低频调制成分 高频响应稳定",
        "包络谱呈单一主峰值 冲击周期固定 无转频相关调制",
    ],
    # 类3: 非等间隔冲击+低频调制 — 间隔不规则、幅值波动、保持架调制
    3: [
        "高频段存在瞬态冲击 冲击间隔变化存在调制现象 非等间隔",
        "时域波形出现脉冲 但间隔不规则 存在低频包络调制",
        "共振频带有响应 冲击幅值波动较大 调制频率低于转频",
        "峭度偏高 存在瞬态成分但周期性弱 冲击间隔随机变化",
        "高频能量波动明显 冲击呈现间歇性 包络谱存在低频调制",
        "脉冲间隔不固定 幅值大小变化明显 存在低频包络调制",
        "频谱共振带响应不规律 冲击间隔随机 调制频率较低",
        "时域波形存在瞬态冲击 但周期性弱 间隔和幅值均有波动",
    ],
    # 类4(扩展): 强周期冲击+深调制(高强度) — 强脉冲、高峭度、深调制
    4: [
        "高频段强烈周期性冲击 冲击能量显著高于一般水平 调制深度增大",
        "时域波形呈现大幅值脉冲 周期性明显 峭度值显著偏高",
        "共振频带能量密集 冲击幅值剧烈 转频调制效应增强 包络谱谐波丰富",
        "信号非高斯性强 瞬态冲击能量集中 高频共振响应剧烈 调制深度深",
        "宽带高频能量分布 冲击间隔与转频相关 幅值波动范围大",
        "周期性冲击能量强烈 调制现象显著 包络谱边带丰富 共振响应剧烈",
        "频谱高频段能量密集 冲击以转频间隔出现 调制深度明显大于一般水平",
        "时域大幅值周期脉冲 峭度极高 调制效应深 高频共振强烈",
    ],
}

from data.text_generator_v3 import generate_unique_texts

# 类别名称（仅用于显示，不输入模型）
CLASS_NAMES = ["正常", "内圈0.007", "外圈0.007", "滚珠0.007", "内圈0.014"]


def get_text_for_label(label, rng=None):
    """随机返回该类别的一条文本变体"""
    variants = TEXT_VARIANTS[label]
    if rng is not None:
        return rng.choice(variants)
    return random.choice(variants)


def get_all_texts():
    """返回所有文本变体的扁平列表（用于穷举评估）"""
    all_texts = []
    for cls in sorted(TEXT_VARIANTS.keys()):
        all_texts.extend(TEXT_VARIANTS[cls])
    return all_texts


# ── CWRU Dataset V2 ───────────────────────────────────────────
class CWRUDatasetV2(Dataset):
    """
    与 V1 相同的数据加载逻辑，但：
    1. 每个样本携带独立的文本描述（随机变体）
    2. __getitem__ 返回 (signal, label, text)
    """

    def __init__(self, root, classes, n_samples_per_class=100,
                 segment_length=512, train=True, train_ratio=0.7, seed=42):
        self.root = root
        self.classes = classes
        self.segment_length = segment_length
        self.train = train
        self.rng = np.random.RandomState(seed)
        self.text_rng = random.Random(seed + 1000)

        label_map = {
            0: 'Normal', 1: 'Inner_0.007', 2: 'Outer_0.007',
            3: 'Ball_0.007', 4: 'Inner_0.014'
        }

        all_segments = []
        all_labels = []

        for cls in classes:
            fault_type = label_map[cls]
            file_ids = CWRU_FILES[fault_type]
            signals = []
            for fid in file_ids:
                path = os.path.join(root, f'{fid}.mat')
                if os.path.exists(path):
                    mat = loadmat(path)
                    key = f'X{fid:03d}_DE_time'
                    if key in mat:
                        sig = mat[key].ravel().astype(np.float32)
                        signals.append(sig)
                    else:
                        print(f"  ⚠️ {key} not in {fid}.mat, keys={[k for k in mat if not k.startswith('__')]}")
                else:
                    print(f"  ⚠️ {path} not found")

            if not signals:
                raise FileNotFoundError(f"No CWRU files found for class {cls} ({fault_type})")

            full_signal = np.concatenate(signals)
            full_signal = (full_signal - full_signal.mean()) / (full_signal.std() + 1e-8)

            n_total = len(full_signal)
            step = segment_length // 2
            max_segments = (n_total - segment_length) // step + 1
            n_needed = n_samples_per_class * 2

            n_select = min(n_needed, max_segments)
            starts = self.rng.choice(max_segments, size=n_select, replace=False) * step

            n_train = int(len(starts) * train_ratio)
            if self.train:
                sel_starts = starts[:n_train]
            else:
                sel_starts = starts[n_train:]

            for s in sel_starts:
                seg = full_signal[s:s + segment_length]
                if len(seg) == segment_length:
                    all_segments.append(seg)
                    all_labels.append(cls)

        self.signals = np.array(all_segments, dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.int64)

        # 为每个样本生成唯一文本（通过模板组合爆炸，组合空间~9万/类）
        self.texts = generate_unique_texts(self.labels.tolist(), seed=seed + 1000)

        print(f"  CWRU V2 数据集: {len(self.signals)} 样本, {len(set(self.labels))} 类")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).unsqueeze(-1)  # (seg_len, 1)
        label = self.labels[idx]
        text = self.texts[idx]
        return signal, label, text


# ── DataLoader 工厂 ──────────────────────────────────────────
def create_dataloaders_v2(root, n_samples_per_class=100, n_classes=4,
                          batch_size=8, train_ratio=0.7,
                          segment_length=512, tokenizer=None, seed=42):
    classes = list(range(n_classes))
    train_ds = CWRUDatasetV2(root, classes, n_samples_per_class,
                            segment_length, train=True, train_ratio=train_ratio, seed=seed)
    test_ds = CWRUDatasetV2(root, classes, n_samples_per_class,
                           segment_length, train=False, train_ratio=train_ratio, seed=seed)

    collate = partial(collate_fn_v2, tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate)
    return train_loader, test_loader, train_ds


def create_extension_dataloader_v2(root, new_classes, n_samples_per_class=50,
                                   batch_size=8, segment_length=512,
                                   tokenizer=None, seed=42):
    ext_ds = CWRUDatasetV2(root, new_classes, n_samples_per_class,
                          segment_length, train=True, train_ratio=1.0, seed=seed + 100)
    collate = partial(collate_fn_v2, tokenizer=tokenizer)
    ext_loader = DataLoader(ext_ds, batch_size=batch_size, shuffle=True,
                           collate_fn=collate)
    return ext_loader, ext_ds


def collate_fn_v2(batch, tokenizer=None, max_text_len=64):
    """批处理：信号 + 标签 + 文本 token化"""
    signals = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    texts = [b[2] for b in batch]

    if tokenizer is not None:
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        return signals, labels, encoded["input_ids"], encoded["attention_mask"]
    return signals, labels, texts


if __name__ == '__main__':
    root = os.path.expanduser('~/tgst_project/datasets/CWRU')
    ds = CWRUDatasetV2(root, [0, 1, 2, 3], n_samples_per_class=100, segment_length=512)
    sig, label, text = ds[0]
    print(f"Signal: {sig.shape}, Label: {label}, Text: {text}")
    print(f"\n各类文本变体示例:")
    for cls in range(5):
        print(f"\n[类{cls} - {CLASS_NAMES[cls]}]")
        for i, t in enumerate(TEXT_VARIANTS[cls]):
            print(f"  变体{i+1}: {t}")
