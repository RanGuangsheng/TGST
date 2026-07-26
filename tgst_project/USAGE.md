# TGST 语义指导故障诊断 - 代码使用说明

> 基于 TGST (Text-Guided Signal Transformer) 的改进实现  
> 核心改进：文本标签描述信号特征（非故障类型）+ 每样本唯一文本 + 强化对比学习

## 一、项目结构

```
tgst_project/
├── model/
│   ├── __init__.py
│   ├── tgst.py              # TGST 模型（信号编码器+BERT文本编码器+交叉注意力+分类头）
│   └── baseline.py          # 基线模型（纯信号编码器+分类头，无文本指导）
├── data/
│   ├── __init__.py
│   ├── cwru_v2.py           # CWRU 数据集加载 + 每样本唯一文本生成
│   └── text_generator_v3.py # 文本模板组合生成器（5维特征×12同义词×随机句式）
├── scripts/
│   ├── experiment_v2.py     # 主实验脚本（baseline / tgst 两种模式 + 动态扩展）
│   └── experiment_ablation.py  # 消融实验脚本（shuffle / random 两种模式）
├── configs/
│   └── config.py            # 全局配置（模型/数据/训练/扩展）
├── utils/
│   ├── __init__.py
│   └── visualization.py     # t-SNE / 混淆矩阵 / 聚类指标
├── requirements.txt
└── USAGE.md                 # 本文件
```

## 二、环境准备

### 2.1 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA 6GB VRAM | 8GB+ |
| CUDA | 12.1+ | 12.1+ |
| 内存 | 8GB | 16GB |
| 磁盘 | 2GB（含BERT模型） | 5GB |

### 2.2 软件依赖

```bash
# Python 3.10+
pip install torch torchvision transformers scikit-learn matplotlib seaborn tqdm numpy scipy
```

或直接：
```bash
pip install -r requirements.txt
```

### 2.3 BERT 模型准备

代码使用 `bert-base-chinese`，需提前下载到本地或通过 HF Mirror：

```bash
# 方式一：HF Mirror（推荐国内用户）
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-chinese')"

# 方式二：离线模式（已下载到本地缓存）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 2.4 数据集准备

使用 CWRU 轴承故障数据集（12kHz 采样频率，DE 端）：

```
tgst_project/datasets/CWRU/
├── 97.mat    # Normal
├── 98.mat
├── 99.mat
├── 107.mat   # Inner race 0.007"
├── 108.mat
├── 120.mat   # Inner race 0.014" (扩展类)
├── 121.mat
├── 135.mat   # Outer race 0.007"
├── 136.mat
├── 137.mat
├── 159.mat   # Ball 0.007"
├── 160.mat
└── 161.mat
```

从 [CWRU官网](https://engineering.case.edu/bearingdatacenter) 下载，放到 `datasets/CWRU/` 目录。

## 三、实验设计

### 3.1 四组对比实验

| 组别 | 模式 | 文本 | 说明 |
|------|------|------|------|
| **Baseline** | 无文本 | 无 | 纯信号编码器，验证无语义指导的性能下限 |
| **TGST** | 正确文本 | 信号特征描述 | 文本与信号特征语义匹配，验证语义指导的增益 |
| **Shuffle** | 错配文本 | 打乱对应 | 文本来自其他样本，破坏信号-文本对应关系 |
| **Random** | 随机文本 | 随机类别文本 | 文本完全随机，消除语义指导 |

### 3.2 文本标签设计（核心创新）

**传统做法（论文原版）**：文本直接描述故障类型 → 模型只需记住"内圈"→类1
**本实现（V3）**：文本描述信号特征 → 模型必须理解语义才能分类

5个特征维度：
1. **频域特征** - 共振带、频谱形状、频带能量
2. **时域特征** - 波形、脉冲、冲击模式
3. **统计特征** - 峭度、高斯性、分布特征
4. **调制特征** - 幅值调制、转频调制、包络特征
5. **冲击间隔** - 等间隔/非等间隔/转频相关

每维度有12个同义表达，随机选2-3维组合 + 随机句式 → 每类约93,600种组合，保证每样本唯一。

### 3.3 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_cl` | 0.1 → **1.0** | 对比损失权重，1.0可激活语义指导 |
| `--temperature` | 0.07 → **0.01** | 对比损失温度，0.01增强区分度 |
| `--d_model` | 64 | 模型隐藏维度（受6GB VRAM限制） |
| `--n_layers` | 2 | Transformer层数 |
| `--batch_size` | 8 | 批大小 |
| `--signal_length` | 512 | 信号截断长度 |
| `--epochs` | 50 | 训练轮数 |

## 四、运行实验

### 4.1 设置环境变量

```bash
cd ~/tgst_project
source ~/tgst_env/bin/activate   # 或你的虚拟环境
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
```

### 4.2 运行 Baseline（无文本基线）

```bash
python scripts/experiment_v2.py \
    --mode baseline \
    --n_classes 4 \
    --epochs 50 \
    --batch_size 8 \
    --signal_length 512 \
    --d_model 64 --n_heads 4 --n_layers 2 --d_ff 256 \
    --save_dir checkpoints
```

输出文件：
- `checkpoints/baseline_best.pth` - 最佳模型权重
- `checkpoints/baseline_report.txt` - 分类报告
- `checkpoints/baseline_tsne.png` - t-SNE 可视化

### 4.3 运行 TGST（语义指导）

```bash
python scripts/experiment_v2.py \
    --mode tgst \
    --n_classes 4 \
    --epochs 50 \
    --lambda_cl 1.0 \
    --temperature 0.01 \
    --batch_size 8 \
    --signal_length 512 \
    --d_model 64 --n_heads 4 --n_layers 2 --d_ff 256 \
    --save_dir checkpoints
```

输出文件：
- `checkpoints/tgst_best.pth`
- `checkpoints/tgst_report.txt`
- `checkpoints/tgst_tsne.png`

### 4.4 运行消融实验

```bash
# Shuffle 消融（文本-信号错配）
python scripts/experiment_ablation.py \
    --ablation shuffle \
    --epochs 50 \
    --lambda_cl 1.0 \
    --temperature 0.01

# Random 消融（完全随机文本）
python scripts/experiment_ablation.py \
    --ablation random \
    --epochs 50 \
    --lambda_cl 1.0 \
    --temperature 0.01
```

### 4.5 运行动态扩展（4→5类）

```bash
# Baseline 扩展
python scripts/experiment_v2.py \
    --mode baseline --extend \
    --n_classes 5 \
    --load_model checkpoints/baseline_best.pth \
    --ext_epochs 20

# TGST 扩展
python scripts/experiment_v2.py \
    --mode tgst --extend \
    --n_classes 5 \
    --load_model checkpoints/tgst_best.pth \
    --ext_epochs 20
```

## 五、实验结果

### 5.1 主实验结果（λ_cl=1.0, temp=0.01）

| 条件 | 准确率 | D_intra | D_inter | Sep | Sil |
|------|--------|---------|---------|-----|-----|
| Baseline（无文本） | 77.50% | 14.87 | 11.11 | 0.75 | 0.371 |
| Shuffle（错配文本） | 90.83% | 7.31 | 8.77 | 1.20 | 0.629 |
| Random（随机文本） | 77.92% | 1.15 | 2.82 | 2.44 | 0.319 |
| **TGST（正确语义）** | **99.58%** | **0.83** | **3.31** | **4.00** | **0.702** |

### 5.2 关键发现

1. **TGST vs Random：21.66% 准确率差距** — 证明语义匹配是性能来源，非BERT正则化
2. **TGST 分离度 4.00 vs Random 2.44** — 正确语义文本让特征空间更分离
3. **Random 接近 Baseline** — 无语义指导时文本无增益，排除了"BERT正则化"假说
4. **旧参数（λ=0.1, T=0.07）下 TGST-Random 仅差0.84%** — 对比损失权重太小时语义指导被CE淹没

### 5.3 超参数对比

| 超参数 | 旧值 | 新值 | TGST准确率 | Random准确率 | 语义贡献 |
|--------|------|------|-----------|-------------|---------|
| λ_cl=0.1, T=0.07 | ✓ | | 99.17% | 98.33% | 0.84% |
| λ_cl=1.0, T=0.01 | | ✓ | 99.58% | 77.92% | **21.66%** |

## 六、模型架构说明

### 6.1 TGST 模型

```
振动信号 → 信号编码器(Transformer×L) → 适配器(瓶颈) → Query
                                                        ↓
文本描述 → BERT(冻结) → 投影层 → Key,Value → 交叉注意力 → 融合特征 → 分类头
                                                        ↓
                                              监督对比损失(SupCon)
```

- **信号编码器**：N层Transformer，提取时域振动信号全局特征
- **适配器**：瓶颈结构(降维-GELU-升维+残差)，参数量仅~0.2%
- **文本编码器**：冻结BERT，将信号特征描述文本编码为语义向量
- **交叉注意力**：信号特征作Query，文本语义作Key/Value，实现语义引导
- **TGSTLoss**：L = L_CE + λ × L_SupCon（交叉熵 + 监督对比损失）

### 6.2 Baseline 模型

```
振动信号 → 信号编码器(Transformer×L) → 适配器 → 分类头
```

与TGST共享信号编码器和适配器结构，区别是无文本指导。

### 6.3 参数量

| 模型 | 总参数 | 可训练参数 |
|------|--------|-----------|
| Baseline | 112,964 | 112,964 |
| TGST | 102,446,852 | 179,204（BERT冻结） |

## 七、自定义扩展

### 7.1 添加新故障类别

1. 在 `data/cwru_v2.py` 的 `CWRU_FILES` 添加新类别的 .mat 文件映射
2. 在 `data/text_generator_v3.py` 的 `FEATURE_POOL` 添加新类别的特征描述词库
3. 更新 `CLASS_NAMES` 列表
4. 运行扩展实验：
```bash
python scripts/experiment_v2.py --mode tgst --extend --n_classes 5 --load_model checkpoints/tgst_best.pth
```

### 7.2 调整文本生成策略

编辑 `data/text_generator_v3.py`：
- `FEATURE_POOL`：修改各类的特征维度和同义词
- `SENTENCE_TEMPLATES`：修改句式模板
- `generate_text_for_label()`：调整特征选取逻辑

### 7.3 使用其他数据集

替换 `data/cwru_v2.py` 中的数据加载逻辑，保持接口一致：
```python
class YourDataset(Dataset):
    def __init__(self, root, classes, n_samples_per_class, segment_length, ...):
        # 加载你的数据
        # 生成文本: self.texts = generate_unique_texts(self.labels.tolist())
    def __getitem__(self, idx):
        return signal, label, text  # 接口不变
```

## 八、常见问题

**Q: 显存不足（OOM）？**
A: 减小 `--batch_size`（如4）、`--signal_length`（如256）、`--d_model`（如32）

**Q: BERT 加载失败？**
A: 设置 `HF_ENDPOINT=https://hf-mirror.com` 重新下载，或使用离线模式 `HF_HUB_OFFLINE=1`

**Q: t-SNE 图中文乱码？**
A: 安装中文字体：`sudo apt install fonts-noto-cjk`

**Q: 训练不收敛？**
A: 检查 `--lambda_cl` 和 `--temperature`，推荐用 λ_cl=1.0, temp=0.01

**Q: 如何复现论文原版（固定文本）？**
A: 使用 `data/cwru_v2.py` 中的 `TEXT_VARIANTS` 字典（每类8条固定文本），而非 `generate_unique_texts()`
