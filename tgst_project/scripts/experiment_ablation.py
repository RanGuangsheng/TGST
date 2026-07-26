"""
TGST V2 消融实验脚本

消融设计：
1. shuffle: 信号和文本随机错配 - 破坏语义对应关系
2. random:  每样本从全部类别文本池随机抽取 - 完全无语义指导

对照组（已完成）：
- baseline:  无文本 -> 77.50%
- tgst:      语义文本 -> 99.17%

预期：
- shuffle: 准确率显著下降（接近 baseline），证明语义对应的重要性
- random:  准确率接近 baseline，证明无语义指导时文本无增益

运行：
  cd ~/tgst_project
  source ~/tgst_env/bin/activate
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

  python scripts/experiment_ablation.py --ablation shuffle --epochs 50
  python scripts/experiment_ablation.py --ablation random --epochs 50
"""
import os
import sys
import time
import argparse
import random as stdlib_random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tgst import TGST, TGSTLoss
from data.cwru_v2 import (
    CWRUDatasetV2,
    create_dataloaders_v2,
    CLASS_NAMES,
)
from data.text_generator_v3 import generate_unique_texts, generate_text_for_label


def get_tokenizer(model_name="bert-base-chinese"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def collate_v2(batch, tokenizer=None, max_text_len=64):
    signals = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    texts = [b[2] for b in batch]
    if tokenizer is not None:
        encoded = tokenizer(texts, padding="max_length", truncation=True,
                           max_length=max_text_len, return_tensors="pt")
        return signals, labels, encoded["input_ids"], encoded["attention_mask"]
    return signals, labels, texts


def apply_shuffle_ablation(train_ds, test_ds, seed=42):
    """
    Shuffle消融：打乱训练集和测试集中文本与信号的对应关系。
    每个样本的文本被随机替换为另一个样本的文本。
    确保打乱后没有样本保留原始文本。
    """
    rng = stdlib_random.Random(seed)

    for ds in [train_ds, test_ds]:
        n = len(ds.texts)
        original_texts = ds.texts[:]
        
        # Fisher-Yates shuffle 直到没有位置保持不变
        while True:
            indices = list(range(n))
            rng.shuffle(indices)
            if all(indices[i] != i for i in range(n)):
                break
        
        ds.texts = [original_texts[indices[i]] for i in range(n)]
    
    return train_ds, test_ds


def apply_random_ablation(train_ds, test_ds, seed=42):
    """
    Random消融：每个样本的文本从全部类别的文本池中随机抽取。
    完全消除文本与信号特征之间的语义对应关系。
    """
    rng = stdlib_random.Random(seed)
    
    for ds in [train_ds, test_ds]:
        n = len(ds.labels)
        new_texts = []
        for i in range(n):
            # 随机选一个类别（不限于当前样本的类别）
            random_cls = rng.choice(range(5))
            text = generate_text_for_label(random_cls, rng)
            new_texts.append(text)
        ds.texts = new_texts
    
    return train_ds, test_ds


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_ce, total_cl, correct, total = 0, 0, 0, 0, 0

    for batch in tqdm(loader, desc="Train"):
        signals, labels, input_ids, attn_mask = batch
        signals, labels = signals.to(device), labels.to(device)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        logits, features = model(signals, input_ids, attn_mask)

        loss, ce, cl = criterion(logits, features, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_cl += cl.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / len(loader),
        "ce": total_ce / len(loader),
        "cl": total_cl / len(loader),
        "acc": correct / total,
    }


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_feats = [], [], []

    for batch in tqdm(loader, desc="Eval"):
        signals, labels, input_ids, attn_mask = batch
        signals, labels = signals.to(device), labels.to(device)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        logits, features = model(signals, input_ids, attn_mask)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_feats.append(features.cpu())

    all_feats = torch.cat(all_feats, 0)
    return correct / total, all_preds, all_labels, all_feats


def compute_clustering_metrics(features, labels):
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    feats_np = features.numpy()
    labels_np = np.array(labels)

    unique_labels = np.unique(labels_np)
    intra_dists = []
    for l in unique_labels:
        mask = labels_np == l
        if mask.sum() > 1:
            center = feats_np[mask].mean(0)
            intra_dists.append(((feats_np[mask] - center) ** 2).sum(1).mean())
    d_intra = np.mean(intra_dists) if intra_dists else 0

    centers = []
    for l in unique_labels:
        mask = labels_np == l
        if mask.sum() > 0:
            centers.append(feats_np[mask].mean(0))
    centers = np.array(centers)

    if len(centers) > 1:
        d_inter = cdist(centers, centers).mean()
    else:
        d_inter = 0

    if len(unique_labels) > 1:
        sil = silhouette_score(feats_np, labels_np)
    else:
        sil = 0

    sep_ratio = d_inter / (d_intra + 1e-8)
    return d_intra, d_inter, sep_ratio, sil


def run_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"TGST-{args.ablation}"

    print(f"\n{'='*60}")
    print(f"  {tag} 消融实验 (V3: 每样本唯一文本)")
    print(f"{'='*60}")
    print(f"🖥️  设备: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Tokenizer
    print("\n📝 加载 BERT tokenizer...")
    tokenizer = get_tokenizer(args.bert_model)

    # 数据
    print("\n📦 准备数据集...")
    data_root = os.path.expanduser('~/tgst_project/datasets/CWRU')
    train_loader, test_loader, train_ds = create_dataloaders_v2(
        root=data_root,
        n_samples_per_class=args.n_samples,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        train_ratio=0.7,
        segment_length=args.signal_length,
        tokenizer=tokenizer,
        seed=42,
    )
    test_ds = test_loader.dataset
    print(f"   训练集: {len(train_ds)} 样本")
    print(f"   测试集: {len(test_ds)} 样本")

    # 应用消融
    print(f"\n🔀 应用消融: {args.ablation}")
    if args.ablation == "shuffle":
        train_ds, test_ds = apply_shuffle_ablation(train_ds, test_ds, seed=42)
        print("   ✓ 打乱验证: 文本-信号对应已随机化")
    elif args.ablation == "random":
        train_ds, test_ds = apply_random_ablation(train_ds, test_ds, seed=42)
        print("   ✓ 随机文本验证: 文本来自所有类别，与信号类别无对应")

    # 重建 DataLoader
    collate = partial(collate_v2, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    # 模型
    n_cls = args.n_classes
    model = TGST(
        signal_dim=1, d_model=args.d_model, n_heads=args.n_heads,
        n_encoder_layers=args.n_layers, d_ff=args.d_ff,
        dropout=args.dropout, n_classes=n_cls, adapter_dim=args.adapter_dim,
        bert_model=args.bert_model, device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  {tag} 模型: 总参数 {n_params:,}, 可训练 {n_trainable:,}")

    optimizer = AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = TGSTLoss(temperature=args.temperature, lambda_cl=args.lambda_cl)

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    save_prefix = f"tgst_{args.ablation}"

    print(f"\n🚀 开始训练 ({args.epochs} epochs)...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        m = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        test_acc, _, _, _ = evaluate_model(model, test_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | Loss: {m['loss']:.4f} "
              f"(CE: {m['ce']:.4f}, CL: {m['cl']:.4f}) | "
              f"Train: {100*m['acc']:.1f}% | Test: {100*test_acc:.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "test_acc": test_acc, "args": vars(args),
            }, os.path.join(args.save_dir, f"{save_prefix}_best.pth"))

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs, "test_acc": test_acc, "args": vars(args),
    }, os.path.join(args.save_dir, f"{save_prefix}_final.pth"))

    # 最终评估
    print(f"\n📊 最终评估...")
    test_acc, preds, labels, feats = evaluate_model(model, test_loader, device)
    d_intra, d_inter, sep, sil = compute_clustering_metrics(feats, labels)

    from sklearn.metrics import classification_report
    report = classification_report(labels, preds, target_names=[CLASS_NAMES[i] for i in range(n_cls)], digits=4)

    print(f"\n{'='*50}")
    print(f"📈 {tag} 消融结果:")
    print(f"   测试准确率: {100*test_acc:.2f}%")
    print(f"   簇内距离: {d_intra:.4f}")
    print(f"   簇间距离: {d_inter:.4f}")
    print(f"   分离度: {sep:.2f}")
    print(f"   Silhouette: {sil:.4f}")
    print(f"\n{report}")
    print(f"{'='*50}")

    # 保存报告
    report_path = os.path.join(args.save_dir, f"{save_prefix}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"{tag} Ablation Experiment (V3)\n")
        f.write(f"Ablation type: {args.ablation}\n")
        f.write(f"Model params: {n_params:,} (trainable: {n_trainable:,})\n")
        f.write(f"Test accuracy: {100*test_acc:.2f}%\n")
        f.write(f"D_intra: {d_intra:.4f}, D_inter: {d_inter:.4f}, Sep: {sep:.2f}, Sil: {sil:.4f}\n\n")
        f.write(report)
    print(f"📁 报告保存: {report_path}")

    # t-SNE
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats) - 1))
        feats_2d = tsne.fit_transform(feats.numpy())

        plt.figure(figsize=(8, 6))
        for i in range(n_cls):
            mask = np.array(labels) == i
            plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                       label=CLASS_NAMES[i], alpha=0.6, s=20)
        plt.title(f"{tag} t-SNE (Ablation)")
        plt.legend()
        plt.tight_layout()
        tsne_path = os.path.join(args.save_dir, f"{save_prefix}_tsne.png")
        plt.savefig(tsne_path, dpi=150)
        plt.close()
        print(f"📁 t-SNE 保存: {tsne_path}")
    except Exception as e:
        print(f"⚠️ t-SNE 失败: {e}")

    return model


def main():
    p = argparse.ArgumentParser(description="TGST V2 Ablation Experiment")
    p.add_argument("--ablation", choices=["shuffle", "random"], required=True,
                   help="消融类型: shuffle=文本打乱, random=随机文本")
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--signal_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_cl", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--adapter_dim", type=int, default=64)
    p.add_argument("--bert_model", type=str, default="bert-base-chinese")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    args = p.parse_args()

    run_ablation(args)


if __name__ == "__main__":
    main()
