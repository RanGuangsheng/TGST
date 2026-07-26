"""
BaselineSignalModel - 无语义指导的基线模型

结构：信号编码器 -> 适配器 -> 分类头
与 TGST 的区别：没有文本编码器、没有跨模态融合
用于对比验证语义指导是否真正提升了信号特征提取能力
"""
import torch
import torch.nn as nn
from model.tgst import SignalEncoder, Adapter


class BaselineModel(nn.Module):
    """
    基线模型：纯信号编码 + 分类
    与 TGST 共享 SignalEncoder + Adapter 结构，区别是无文本指导
    """

    def __init__(
        self,
        signal_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 4,
        adapter_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # 信号编码器（与TGST相同）
        self.signal_encoder = SignalEncoder(
            signal_dim=signal_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 适配器（与TGST相同）
        self.adapter = Adapter(d_model, adapter_dim)

        # 分类头（与TGST结构相同，但输入是纯信号特征）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, signal, *args, **kwargs):
        """
        Args:
            signal: (B, N, signal_dim)
        Returns:
            logits: (B, n_classes)
            features: (B, d_model) 信号特征（用于对比学习）
        """
        feat = self.signal_encoder(signal)  # (B, d_model)
        feat = self.adapter(feat)            # (B, d_model)
        logits = self.classifier(feat)
        return logits, feat

    def freeze_for_extension(self):
        """冻结除适配器外所有参数（与TGST一致）"""
        for name, param in self.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Baseline Extension] 可训练参数: {trainable}/{total} ({100*trainable/total:.1f}%)")

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
