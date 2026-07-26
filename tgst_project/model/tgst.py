# TGST: Text-guided Signal Transformer
# 旋转机械故障诊断 - 基于语义指导和动态扩展的复合故障诊断模型框架
# 论文: 10.3901/JME.260770

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """标准正弦余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q: (B, Nq, d), K: (B, Nk, d), V: (B, Nk, d)
        B, Nq, _ = Q.shape
        Nk = K.size(1)

        Q = self.W_q(Q).view(B, Nq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, Nk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, Nk, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, h, Nq, Nk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, h, Nq, d_k)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_model)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """标准 Transformer 编码器层"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN style
        normed = self.norm1(x)
        x = x + self.dropout1(self.mha(normed, normed, normed))
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))
        return x


class SignalEncoder(nn.Module):
    """
    信号编码器：将一维时域振动信号编码为特征向量
    - 输入: (B, N, signal_dim) 其中 N 为时间序列长度, signal_dim 为信号维度
    - 输出: (B, d_model) 全局特征向量
    """

    def __init__(
        self,
        signal_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.signal_proj = nn.Linear(signal_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N, signal_dim)
        x = self.signal_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # 全局平均池化 -> (B, d_model)
        x = x.mean(dim=1)
        return x


class Adapter(nn.Module):
    """
    适配器：瓶颈结构，将信号特征映射至文本嵌入空间
    降维 -> GELU -> 升维 + 残差连接
    参数量仅约 2*d*r，远小于全模型
    """

    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.gelu = nn.GELU()
        # 初始化为接近零，保证初始时不破坏原特征
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x: (B, d_model)
        h = self.down(x)
        h = self.gelu(h)
        h = self.up(h)
        return x + h  # 残差连接


class TextEncoder(nn.Module):
    """
    文本编码器：使用预训练 BERT 将文本指令转化为语义向量
    论文核心：利用预训练语言模型的语义知识指导信号特征学习
    BERT 参数冻结，仅训练投影层
    """

    def __init__(self, model_name: str = "bert-base-chinese", d_model: int = 256, freeze: bool = True):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden = self.bert.config.hidden_size  # 768

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 投影层: BERT 768 -> d_model (可训练)
        self.proj = nn.Linear(self.bert_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, T), attention_mask: (B, T)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (B, T, 768)

        # 投影到 d_model
        hidden = self.proj(hidden_states)  # (B, T, d_model)
        hidden = self.norm(hidden)
        return hidden


class CrossAttention(nn.Module):
    """
    跨模态融合层：交叉注意力
    Q = 信号特征 (Query)
    K, V = 文本语义 (Key, Value)
    动态建立信号特征与文本描述的语义关联
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        # Q: (B, d_model) -> (B, 1, d_model)
        # K: (B, T, d_model), V: (B, T, d_model)
        if Q.dim() == 2:
            Q = Q.unsqueeze(1)  # (B, 1, d_model)

        Q_norm = self.norm_q(Q)
        out = self.attn(Q_norm, K, V)  # (B, 1, d_model)
        out = self.dropout(out)
        # 残差连接 + 层归一化
        out = self.norm_out(Q + out)
        return out.squeeze(1)  # (B, d_model)


class TGST(nn.Module):
    """
    TGST: Text-guided Signal Transformer
    完整的复合故障诊断模型框架

    结构：
    1. 信号编码器 -> 适配器 -> (B, d_model)
    2. 文本编码器 (BERT, 冻结) -> (B, T, d_model)
    3. 交叉注意力融合 -> (B, d_model)
    4. 分类头 -> (B, num_classes)
    """

    def __init__(
        self,
        signal_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 12,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 8,
        adapter_dim: int = 64,
        bert_model: str = "bert-base-chinese",
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.device = device

        # 1. 信号编码器
        self.signal_encoder = SignalEncoder(
            signal_dim=signal_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 2. 适配器
        self.adapter = Adapter(d_model, adapter_dim)

        # 3. 文本编码器 (预训练BERT, 冻结)
        self.text_encoder = TextEncoder(bert_model, d_model, freeze=True)

        # 4. 跨模态融合
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)

        # 5. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, signal, input_ids, attention_mask=None):
        """
        Args:
            signal: (B, N, signal_dim) 振动信号
            input_ids: (B, T) 文本token
            attention_mask: (B, T) 注意力mask
        Returns:
            logits: (B, n_classes)
            features: (B, d_model) 融合特征 (用于对比学习)
        """
        # 信号编码
        sig_feat = self.signal_encoder(signal)  # (B, d_model)
        sig_feat = self.adapter(sig_feat)       # (B, d_model)

        # 文本编码
        text_feat = self.text_encoder(input_ids, attention_mask)  # (B, T, d_model)

        # 跨模态融合
        fused = self.cross_attn(sig_feat, text_feat, text_feat)  # (B, d_model)

        # 分类
        logits = self.classifier(fused)
        return logits, fused

    def freeze_for_extension(self):
        """
        冻结除适配器外的所有参数，用于动态扩展
        仅适配器可训练 -> 实现低成本故障类型扩展
        """
        for name, param in self.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Dynamic Extension] 可训练参数: {trainable}/{total} ({100*trainable/total:.1f}%)")

    def get_trainable_params(self):
        """获取当前可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]


class TGSTLoss(nn.Module):
    """
    组合损失函数: L = L_CE + λ * L_CL
    L_CE: 交叉熵分类损失
    L_CL: 对比损失 (SupCon), 拉近同类特征、推开异类特征
    """

    def __init__(self, temperature: float = 0.07, lambda_cl: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.lambda_cl = lambda_cl

    def forward(self, logits, features, labels):
        # 交叉熵损失
        loss_ce = self.ce(logits, labels)

        # 监督对比损失 (SupCon)
        loss_cl = self._supcon_loss(features, labels)
        return loss_ce + self.lambda_cl * loss_cl, loss_ce, loss_cl

    def _supcon_loss(self, features, labels):
        """
        Supervised Contrastive Loss
        features: (B, d)
        labels: (B,)
        """
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # mask 对角线
        mask = torch.eye(B, device=features.device).bool()
        sim = sim.masked_fill(mask, -1e9)

        # 同类mask
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        label_eq = label_eq & ~mask  # 排除自身

        # 没有正样本时跳过
        if label_eq.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # log-softmax over all negatives + positives
        log_prob = F.log_softmax(sim, dim=1)

        # 平均正样本的 log_prob
        pos_log_prob = (log_prob * label_eq.float()).sum(dim=1)
        pos_count = label_eq.float().sum(dim=1).clamp(min=1.0)
        loss = -(pos_log_prob / pos_count).mean()
        return loss
