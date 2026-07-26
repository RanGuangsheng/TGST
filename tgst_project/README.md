# TGST: Text-Guided Signal Transformer

> 基于语义指导和动态扩展的轴承故障诊断框架  
> 改进自: DOI 10.3901/JME.260770 (机械工程学报 2026, 62(11))

## 核心改进

相比原论文的三个关键改进：

1. **文本标签描述信号特征（非故障类型）** — 文本描述频域/时域/统计/调制/频带能量特征，不含"内圈""外圈"等故障类型词，迫使模型真正理解语义而非记忆关键词
2. **每样本唯一文本** — 通过模板组合爆炸（5维×12同义词×随机句式，~93,600组合/类），保证每个样本文本唯一，杜绝文本→标签记忆
3. **强化对比学习** — λ_cl=1.0, temp=0.01（原0.1/0.07），激活语义指导，TGST-Random准确率差距从0.84%提升至21.66%

## 项目结构

```
tgst_project/
├── model/
│   ├── tgst.py              # TGST模型（信号编码器+BERT+交叉注意力+Adapter+SupCon损失）
│   └── baseline.py          # 基线模型（纯信号编码器，无文本指导）
├── data/
│   ├── cwru_v2.py           # CWRU数据集加载 + 每样本唯一文本生成
│   └── text_generator_v3.py # 文本模板组合生成器（5维特征×12同义词×随机句式）
├── scripts/
│   ├── experiment_v2.py     # 主实验脚本（baseline/tgst + 动态扩展4→5类）
│   └── experiment_ablation.py  # 消融实验脚本（shuffle/random）
├── configs/
│   └── config.py            # 全局配置
├── utils/
│   └── visualization.py     # t-SNE/混淆矩阵/聚类指标
├── requirements.txt
├── USAGE.md                # 详细运行说明
└── README.md
```

## 数据集

**CWRU 轴承故障数据集**（Case Western Reserve University）

- 采样频率: 12kHz, DE端加速度计
- 载荷: 0HP
- 初始4类: Normal(97-99), Inner_0.007"(107,108), Outer_0.007"(135-137), Ball_0.007"(159-161)
- 扩展类: Inner_0.014"(120,121)
- 560训练 / 240测试样本（4类），信号长度512点

从 [CWRU官网](https://engineering.case.edu/bearingdatacenter) 下载 `.mat` 文件，放到 `datasets/CWRU/` 目录。

## 模型架构

```
振动信号 → 信号编码器(Transformer×2) → Adapter(瓶颈) → Query
                                                      ↓
文本描述 → BERT(冻结) → 投影层 → Key,Value → 交叉注意力 → 融合特征 → 分类头
                                                      ↓
                                            监督对比损失(SupCon)
```

### 关键组件

| 组件 | 说明 |
|------|------|
| SignalEncoder | 2层Transformer，提取振动信号全局特征 |
| Adapter | 瓶颈结构(降维-GELU-升维+残差)，零初始化，参数量仅0.2% |
| TextEncoder | 冻结BERT-base-chinese + 可训练投影层(768→64) |
| CrossAttention | 信号特征作Q，文本语义作K/V，实现语义引导 |
| TGSTLoss | L = L_CE + λ × L_SupCon（交叉熵 + 监督对比损失） |

### 动态扩展（4→5类）

新增故障类型时：
- 冻结全部参数，仅解冻Adapter + 新分类头行
- 迁移旧分类头权重 + 梯度hook冻结旧行
- 回放缓冲(Replay Buffer)混合训练，对抗灾难性遗忘
- 20 epoch内完成扩展

## 实验结果

### 主实验 + 消融（λ_cl=1.0, temp=0.01, 50 epochs）

| 条件 | 准确率 | 分离度Sep | Silhouette |
|------|--------|-----------|------------|
| Baseline（无文本） | 77.50% | 0.75 | 0.371 |
| Random（随机文本） | 77.92% | 2.44 | 0.319 |
| **TGST（正确语义）** | **99.58%** | **4.00** | **0.702** |

- TGST vs Random 准确率差距 **21.66%**，证明语义匹配是性能来源

### 动态扩展（4→5类）

| 指标 | Baseline | TGST |
|------|----------|------|
| 新类准确率 | 91.0% | **97.0%** |
| 旧类准确率 | 52.5% | **93.8%** |
| 全类准确率 | 59.7% | **94.7%** |

- TGST旧类保持93.8% vs Baseline仅52.5%，语义文本作为"知识锚点"有效对抗灾难性遗忘

## 快速开始

```bash
# 环境
cd ~/tgst_project
source ~/tgst_env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# Baseline
python scripts/experiment_v2.py --mode baseline --n_classes 4 --epochs 50

# TGST（语义指导）
python scripts/experiment_v2.py --mode tgst --n_classes 4 --epochs 50 --lambda_cl 1.0 --temperature 0.01

# 消融实验
python scripts/experiment_ablation.py --ablation shuffle --epochs 50 --lambda_cl 1.0 --temperature 0.01
python scripts/experiment_ablation.py --ablation random --epochs 50 --lambda_cl 1.0 --temperature 0.01

# 动态扩展
python scripts/experiment_v2.py --mode tgst --extend --n_classes 5 --load_model checkpoints/tgst_best.pth --ext_epochs 20 --lambda_cl 1.0 --temperature 0.01
```

详细说明见 [USAGE.md](USAGE.md)。

## 硬件要求

- GPU: NVIDIA RTX 2060 (6GB VRAM) 或更高
- CUDA: 12.1+, Python 3.10+
- 依赖: PyTorch 2.1+, transformers, scikit-learn, scipy

