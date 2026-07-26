"""
TGST 可视化工具
- t-SNE 特征可视化
- 混淆矩阵
- 训练曲线
- 类内/类间距离分析
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
from typing import List, Optional, Dict
import os


def plot_tsne(features: np.ndarray, labels: np.ndarray, save_path: str, title: str = "t-SNE"):
    """t-SNE 特征空间可视化"""
    print(f"  计算 t-SNE (n={len(features)})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedded[:, 0], embedded[:, 1],
        c=labels, cmap="tab10", s=15, alpha=0.7, edgecolors="none"
    )
    plt.colorbar(scatter, label="Fault Class")
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 保存: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    """混淆矩阵可视化"""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Accuracy (%)"}
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 保存: {save_path}")


def compute_cluster_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """计算类内/类间距离"""
    unique_labels = np.unique(labels)
    centers = {}
    for label in unique_labels:
        mask = labels == label
        centers[label] = features[mask].mean(axis=0)

    # 类内平均距离
    intra_distances = []
    for label in unique_labels:
        mask = labels == label
        dists = np.linalg.norm(features[mask] - centers[label], axis=1)
        intra_distances.extend(dists)
    d_intra = np.mean(intra_distances)

    # 类间距离
    inter_distances = []
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i + 1:]:
            dist = np.linalg.norm(centers[l1] - centers[l2])
            inter_distances.append(dist)
    d_inter = np.mean(inter_distances) if inter_distances else 0.0

    return {
        "d_intra": d_intra,
        "d_inter": d_inter,
        "separation_ratio": d_inter / (d_intra + 1e-8),
    }


def plot_training_curves(history: List[Dict], save_path: str):
    """训练曲线可视化"""
    epochs = range(1, len(history) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, [h["loss"] for h in history], "b-", label="Total Loss")
    axes[0].plot(epochs, [h["ce"] for h in history], "r--", label="CE Loss")
    axes[0].plot(epochs, [h["cl"] for h in history], "g--", label="CL Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [100 * h["acc"] for h in history], "b-", label="Train Acc")
    if "test_acc" in history[0]:
        axes[1].plot(epochs, [100 * h["test_acc"] for h in history], "r-", label="Test Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    if "lr" in history[0]:
        axes[2].plot(epochs, [h["lr"] for h in history], "b-")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 保存: {save_path}")


@torch.no_grad()
def extract_features(model, loader, device, tokenizer):
    """提取所有样本的融合特征"""
    model.eval()
    all_features = []
    all_labels = []

    for signals, labels, input_ids, attn_mask in loader:
        signals = signals.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        _, features = model(signals, input_ids, attn_mask)
        all_features.append(features.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.concatenate(all_features), np.array(all_labels)
