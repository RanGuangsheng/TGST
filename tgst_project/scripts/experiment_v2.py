"""
TGST V2 统一实验脚本

实验设计：
1. Baseline: 纯信号编码器 -> 分类 (无文本指导)
2. TGST: 信号编码器 + BERT文本编码器 + 跨注意力融合 (语义指导)

两组使用完全相同的：
- 信号编码器结构 (d_model, n_layers, n_heads, d_ff)
- 适配器结构
- 分类头结构
- 训练超参数 (lr, epochs, batch_size, loss)
- 数据集 (CWRU, 信号特征描述型文本标签)
- 随机种子

唯一区别：TGST 有文本编码器 + 跨注意力，Baseline 没有。

运行：
  cd ~/tgst_project
  source ~/tgst_env/bin/activate
  export HF_ENDPOINT=https://hf-mirror.com
  
  # 基线实验
  python scripts/experiment_v2.py --mode baseline --n_classes 4 --epochs 50
  
  # TGST实验  
  python scripts/experiment_v2.py --mode tgst --n_classes 4 --epochs 50
  
  # 扩展实验
  python scripts/experiment_v2.py --mode baseline --extend --n_classes 5 --load_model checkpoints/baseline_best.pth
  python scripts/experiment_v2.py --mode tgst --extend --n_classes 5 --load_model checkpoints/tgst_best.pth
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.tgst import TGST, TGSTLoss
from model.baseline import BaselineModel
from data.cwru_v2 import (
    CWRUDatasetV2,
    create_dataloaders_v2,
    create_extension_dataloader_v2,
    TEXT_VARIANTS,
    CLASS_NAMES,
)


def get_tokenizer(model_name="bert-base-chinese"):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def collate_v2(batch, tokenizer=None, max_text_len=64):
    """统一 collate: 基线模式 tokenizer=None 跳过文本"""
    signals = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    texts = [b[2] for b in batch]

    if tokenizer is not None:
        encoded = tokenizer(texts, padding="max_length", truncation=True,
                           max_length=max_text_len, return_tensors="pt")
        return signals, labels, encoded["input_ids"], encoded["attention_mask"]
    return signals, labels, texts


def train_epoch(model, loader, optimizer, criterion, device, is_tgst):
    model.train()
    total_loss, total_ce, total_cl, correct, total = 0, 0, 0, 0, 0

    for batch in tqdm(loader, desc="Train"):
        if is_tgst:
            signals, labels, input_ids, attn_mask = batch
            signals, labels = signals.to(device), labels.to(device)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            logits, features = model(signals, input_ids, attn_mask)
        else:
            signals, labels, _ = batch
            signals, labels = signals.to(device), labels.to(device)
            logits, features = model(signals)

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
def evaluate_model(model, loader, device, is_tgst):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_feats = [], [], []

    for batch in tqdm(loader, desc="Eval"):
        if is_tgst:
            signals, labels, input_ids, attn_mask = batch
            signals, labels = signals.to(device), labels.to(device)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            logits, features = model(signals, input_ids, attn_mask)
        else:
            signals, labels, _ = batch
            signals, labels = signals.to(device), labels.to(device)
            logits, features = model(signals)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_feats.append(features.cpu())

    all_feats = torch.cat(all_feats, 0)
    return correct / total, all_preds, all_labels, all_feats


def compute_clustering_metrics(features, labels):
    """计算聚类质量指标"""
    from sklearn.metrics import silhouette_score
    feats_np = features.numpy()
    labels_np = np.array(labels)
    
    # 簇内距离 (平均)
    unique_labels = np.unique(labels_np)
    intra_dists = []
    for l in unique_labels:
        mask = labels_np == l
        if mask.sum() > 1:
            center = feats_np[mask].mean(0)
            intra_dists.append(((feats_np[mask] - center) ** 2).sum(1).mean())
    d_intra = np.mean(intra_dists) if intra_dists else 0

    # 簇间距离
    centers = []
    for l in unique_labels:
        mask = labels_np == l
        if mask.sum() > 0:
            centers.append(feats_np[mask].mean(0))
    centers = np.array(centers)
    
    if len(centers) > 1:
        from scipy.spatial.distance import cdist
        d_inter = cdist(centers, centers).mean()
    else:
        d_inter = 0

    # silhouette score
    if len(unique_labels) > 1:
        sil = silhouette_score(feats_np, labels_np)
    else:
        sil = 0

    sep_ratio = d_inter / (d_intra + 1e-8)
    return d_intra, d_inter, sep_ratio, sil


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_tgst = args.mode == "tgst"
    tag = "TGST" if is_tgst else "Baseline"
    
    print(f"\n{'='*60}")
    print(f"  {tag} 实验 (V2: 信号特征描述型文本)")
    print(f"{'='*60}")
    print(f"🖥️  设备: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Tokenizer (仅TGST需要)
    tokenizer = None
    if is_tgst:
        print("\n📝 加载 BERT tokenizer...")
        tokenizer = get_tokenizer(args.bert_model)

    # 数据
    print("\n📦 准备数据集...")
    data_root = os.path.expanduser('~/tgst_project/datasets/CWRU')
    train_loader, test_loader, _ = create_dataloaders_v2(
        root=data_root,
        n_samples_per_class=args.n_samples,
        n_classes=args.n_classes if not args.extend else (args.n_classes - 1),
        batch_size=args.batch_size,
        train_ratio=0.7,
        segment_length=args.signal_length,
        tokenizer=tokenizer,
        seed=42,
    )
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   测试集: {len(test_loader.dataset)} 样本")

    if not args.extend:
        # ── 初始训练 ──────────────────────────────
        n_cls = args.n_classes

        if is_tgst:
            model = TGST(
                signal_dim=1, d_model=args.d_model, n_heads=args.n_heads,
                n_encoder_layers=args.n_layers, d_ff=args.d_ff,
                dropout=args.dropout, n_classes=n_cls, adapter_dim=args.adapter_dim,
                bert_model=args.bert_model, device=device,
            ).to(device)
        else:
            model = BaselineModel(
                signal_dim=1, d_model=args.d_model, n_heads=args.n_heads,
                n_encoder_layers=args.n_layers, d_ff=args.d_ff,
                dropout=args.dropout, n_classes=n_cls, adapter_dim=args.adapter_dim,
            ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🏗️  {tag} 模型: 总参数 {n_params:,}, 可训练 {n_trainable:,}")

        optimizer = AdamW(model.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        criterion = TGSTLoss(temperature=args.temperature, lambda_cl=args.lambda_cl)

        os.makedirs(args.save_dir, exist_ok=True)
        best_acc = 0.0
        save_prefix = "tgst" if is_tgst else "baseline"

        print(f"\n🚀 开始训练 ({args.epochs} epochs)...")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            m = train_epoch(model, train_loader, optimizer, criterion, device, is_tgst)
            scheduler.step()
            test_acc, _, _, _ = evaluate_model(model, test_loader, device, is_tgst)
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

        # 最终评估 + 聚类指标
        print(f"\n📊 最终评估...")
        test_acc, preds, labels, feats = evaluate_model(model, test_loader, device, is_tgst)
        d_intra, d_inter, sep, sil = compute_clustering_metrics(feats, labels)
        
        from sklearn.metrics import classification_report
        report = classification_report(labels, preds, target_names=[CLASS_NAMES[i] for i in range(n_cls)], digits=4)
        
        print(f"\n{'='*50}")
        print(f"📈 {tag} 训练结果:")
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
            f.write(f"{tag} Experiment (V2)\n")
            f.write(f"Model params: {n_params:,} (trainable: {n_trainable:,})\n")
            f.write(f"Test accuracy: {100*test_acc:.2f}%\n")
            f.write(f"D_intra: {d_intra:.4f}, D_inter: {d_inter:.4f}, Sep: {sep:.2f}, Sil: {sil:.4f}\n\n")
            f.write(report)
        print(f"📁 报告保存: {report_path}")

        # t-SNE 可视化
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
            plt.title(f"{tag} t-SNE (V2)")
            plt.legend()
            plt.tight_layout()
            tsne_path = os.path.join(args.save_dir, f"{save_prefix}_tsne.png")
            plt.savefig(tsne_path, dpi=150)
            plt.close()
            print(f"📁 t-SNE 保存: {tsne_path}")
        except Exception as e:
            print(f"⚠️ t-SNE 失败: {e}")

        return model

    else:
        # ── 动态扩展 ──────────────────────────────
        print(f"\n🔧 动态扩展模式 (4 -> {args.n_classes} 类)")
        
        checkpoint = torch.load(args.load_model, map_location=device, weights_only=False)
        old_args = checkpoint["args"]
        old_n = old_args["n_classes"]

        if is_tgst:
            model = TGST(
                signal_dim=1, d_model=old_args["d_model"], n_heads=old_args["n_heads"],
                n_encoder_layers=old_args["n_layers"], d_ff=old_args["d_ff"],
                dropout=old_args["dropout"], n_classes=args.n_classes, adapter_dim=old_args["adapter_dim"],
                bert_model=old_args["bert_model"], device=device,
            ).to(device)
        else:
            model = BaselineModel(
                signal_dim=1, d_model=old_args["d_model"], n_heads=old_args["n_heads"],
                n_encoder_layers=old_args["n_layers"], d_ff=old_args["d_ff"],
                dropout=old_args["dropout"], n_classes=args.n_classes, adapter_dim=old_args["adapter_dim"],
            ).to(device)

        # 加载旧参数
        state = checkpoint["model_state_dict"]
        model_state = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(loaded)
        model.load_state_dict(model_state)

        # 迁移旧分类头权重
        old_w = state["classifier.3.weight"]
        old_b = state["classifier.3.bias"]
        model.classifier[3].weight.data[:old_n] = old_w
        model.classifier[3].bias.data[:old_n] = old_b
        print(f"   已迁移旧分类头: {old_n} -> {args.n_classes} 类")

        # 冻结 + 解冻
        model.freeze_for_extension()
        model.classifier[3].weight.requires_grad_(True)
        model.classifier[3].bias.requires_grad_(True)

        # 梯度 hook 冻结旧行
        def freeze_old_w(grad):
            grad[:old_n, :] = 0
            return grad
        def freeze_old_b(grad):
            grad[:old_n] = 0
            return grad
        model.classifier[3].weight.register_hook(freeze_old_w)
        model.classifier[3].bias.register_hook(freeze_old_b)

        # 扩展数据
        new_classes = list(range(old_n, args.n_classes))
        ext_loader, _ = create_extension_dataloader_v2(
            root=data_root, new_classes=new_classes,
            n_samples_per_class=args.n_samples, batch_size=args.batch_size,
            segment_length=args.signal_length, tokenizer=tokenizer,
        )

        # 回放缓冲
        old_train_loader, old_test_loader, _ = create_dataloaders_v2(
            root=data_root, n_samples_per_class=args.n_samples,
            n_classes=old_n, batch_size=args.batch_size,
            train_ratio=0.7, segment_length=args.signal_length,
            tokenizer=tokenizer, seed=42,
        )
        replay = []
        for batch in old_train_loader:
            if is_tgst:
                s, l, ii, am = batch
                for i in range(s.size(0)):
                    replay.append((s[i], l[i].item(), ii[i], am[i]))
            else:
                s, l, t = batch
                for i in range(s.size(0)):
                    replay.append((s[i], l[i].item(), t[i]))

        # 混合数据集
        class MixedDataset(torch.utils.data.Dataset):
            def __init__(self, new_loader, replay):
                self.new_data = []
                for batch in new_loader:
                    if is_tgst:
                        s, l, ii, am = batch
                        for i in range(s.size(0)):
                            self.new_data.append((s[i], l[i].item(), ii[i], am[i]))
                    else:
                        s, l, t = batch
                        for i in range(s.size(0)):
                            self.new_data.append((s[i], l[i].item(), t[i]))
                self.replay = replay
            def __len__(self):
                return len(self.new_data) + len(self.replay)
            def __getitem__(self, idx):
                if idx < len(self.new_data):
                    return self.new_data[idx]
                return self.replay[idx - len(self.new_data)]

        mixed = MixedDataset(ext_loader, replay)
        if is_tgst:
            # 预 tokenize 的数据：直接 stack tensor，不再走 tokenizer
            def collate_pretokenized(batch):
                signals = torch.stack([b[0] for b in batch])
                labels = torch.tensor([b[1] for b in batch])
                input_ids = torch.stack([b[2] for b in batch])
                attn_mask = torch.stack([b[3] for b in batch])
                return signals, labels, input_ids, attn_mask
            mixed_loader = DataLoader(mixed, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=collate_pretokenized)
        else:
            mixed_loader = DataLoader(mixed, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=partial(collate_v2, tokenizer=None))

        print(f"   混合训练集: {len(mixed)} (新类 {len(mixed) - len(replay)} + 回放 {len(replay)})")

        optimizer = AdamW(model.get_trainable_params(), lr=args.ext_lr, weight_decay=1e-4)
        criterion = TGSTLoss(temperature=args.temperature, lambda_cl=args.lambda_cl)

        t0 = time.time()
        for epoch in range(1, args.ext_epochs + 1):
            m = train_epoch(model, mixed_loader, optimizer, criterion, device, is_tgst)
            print(f"Ext {epoch}/{args.ext_epochs} | Loss: {m['loss']:.4f} | Acc: {100*m['acc']:.1f}%")
        elapsed = time.time() - t0

        # 评估
        ext_acc, ext_preds, ext_labels, ext_feats = evaluate_model(model, ext_loader, device, is_tgst)
        old_acc, old_preds, old_labels, old_feats = evaluate_model(model, old_test_loader, device, is_tgst)

        # 全类别测试
        _, full_test_loader, _ = create_dataloaders_v2(
            root=data_root, n_samples_per_class=args.n_samples,
            n_classes=args.n_classes, batch_size=args.batch_size,
            train_ratio=0.7, segment_length=args.signal_length,
            tokenizer=tokenizer, seed=42,
        )
        full_acc, preds, labels, feats = evaluate_model(model, full_test_loader, device, is_tgst)

        # 聚类指标
        d_intra, d_inter, sep, sil = compute_clustering_metrics(feats, labels)

        # 逐类分类报告
        from sklearn.metrics import classification_report
        full_class_names = [CLASS_NAMES[i] for i in range(args.n_classes)]
        report = classification_report(labels, preds, target_names=full_class_names, digits=4)

        save_prefix = "tgst" if is_tgst else "baseline"
        print(f"\n{'='*50}")
        print(f"📈 {tag} 扩展结果 (4 -> {args.n_classes} 类):")
        print(f"   新类准确率: {100*ext_acc:.1f}%")
        print(f"   旧类准确率: {100*old_acc:.1f}%")
        print(f"   全类准确率: {100*full_acc:.1f}%")
        print(f"   簇内距离: {d_intra:.4f}")
        print(f"   簇间距离: {d_inter:.4f}")
        print(f"   分离度: {sep:.2f}")
        print(f"   Silhouette: {sil:.4f}")
        print(f"   扩展耗时: {elapsed:.0f}s")
        print(f"\n{report}")
        print(f"{'='*50}")

        # 保存模型
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_classes": args.n_classes, "ext_acc": ext_acc,
            "old_acc": old_acc, "full_acc": full_acc,
            "d_intra": d_intra, "d_inter": d_inter, "sep": sep, "sil": sil,
            "args": vars(args),
        }, os.path.join(args.save_dir, f"{save_prefix}_extended.pth"))

        # 保存报告
        report_path = os.path.join(args.save_dir, f"{save_prefix}_extended_report.txt")
        with open(report_path, "w") as f:
            f.write(f"{tag} Extension Experiment (4 -> {args.n_classes} classes)\n")
            f.write(f"Data: CWRU 12kHz DE 0HP\n")
            f.write(f"Hyperparams: lambda_cl={args.lambda_cl}, temperature={args.temperature}\n")
            f.write(f"Ext epochs: {args.ext_epochs}, ext_lr: {args.ext_lr}\n\n")
            f.write(f"New class accuracy: {100*ext_acc:.2f}%\n")
            f.write(f"Old class accuracy: {100*old_acc:.2f}%\n")
            f.write(f"Full accuracy: {100*full_acc:.2f}%\n")
            f.write(f"D_intra: {d_intra:.4f}, D_inter: {d_inter:.4f}, Sep: {sep:.2f}, Sil: {sil:.4f}\n\n")
            f.write(report)
        print(f"📁 报告保存: {report_path}")

        # t-SNE 可视化
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feats) - 1))
            feats_2d = tsne.fit_transform(feats.numpy())

            plt.figure(figsize=(8, 6))
            for i in range(args.n_classes):
                mask = np.array(labels) == i
                plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                           label=CLASS_NAMES[i], alpha=0.6, s=20)
            plt.title(f"{tag} t-SNE (Extension {args.n_classes} classes)")
            plt.legend()
            plt.tight_layout()
            tsne_path = os.path.join(args.save_dir, f"{save_prefix}_extended_tsne.png")
            plt.savefig(tsne_path, dpi=150)
            plt.close()
            print(f"📁 t-SNE 保存: {tsne_path}")
        except Exception as e:
            print(f"⚠️ t-SNE 失败: {e}")

        return model


def main():
    p = argparse.ArgumentParser(description="TGST V2 Experiment")
    p.add_argument("--mode", choices=["baseline", "tgst"], required=True)
    p.add_argument("--extend", action="store_true", help="动态扩展模式")
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
    p.add_argument("--ext_epochs", type=int, default=20)
    p.add_argument("--ext_lr", type=float, default=5e-4)
    p.add_argument("--load_model", type=str, default="")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    args = p.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
