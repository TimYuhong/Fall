"""RACA radar fall/activity classifier.

Feature shapes:
    RD : (T=100, range=128, doppler=64)
    RA : (T=100, angle=361, range=128)
    RE : (T=100, angle=361, range=128)

Usage::
    python -m training.train --manifest dataset/train_manifest.jsonl --out-dir checkpoints
"""
from __future__ import annotations
import argparse, json, os, time
from collections import Counter
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RadarManifestDataset(Dataset):
    def __init__(self, manifest_path: str, label_map: Optional[Dict[str,int]]=None):
        self.records: List[Dict] = []
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        all_labels = sorted({r["label"] for r in self.records})
        self.label_map = label_map or {l: i for i, l in enumerate(all_labels)}
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.labels = [self.label_map[r["label"]] for r in self.records]

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rd = torch.from_numpy(np.load(rec["rd_path"]).astype(np.float32))  # (T,R,D)
        ra = torch.from_numpy(np.load(rec["ra_path"]).astype(np.float32))  # (T,A,R)
        re = torch.from_numpy(np.load(rec["re_path"]).astype(np.float32))  # (T,A,R)
        label = torch.tensor(self.label_map[rec["label"]], dtype=torch.long)
        return rd, ra, re, label

def collate_fn(batch):
    rds, ras, res, labels = zip(*batch)
    return torch.stack(rds), torch.stack(ras), torch.stack(res), torch.stack(labels)

# ---------------------------------------------------------------------------
# Model: Range-Anchored Cross-Attention (RACA)
# ---------------------------------------------------------------------------
class RangeAnchoredCrossAttention(nn.Module):
    """Cross-attention anchored at the shared Range dimension."""
    def __init__(self, dim: int, num_heads: int=4, dropout: float=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_query: torch.Tensor, x_key: torch.Tensor) -> torch.Tensor:
        # x_query: (B, C, R, Dq)  x_key: (B, C, R, Dk)
        B, C, R, Dq = x_query.shape
        Dk = x_key.shape[3]
        H, hd = self.num_heads, C // self.num_heads
        # merge B*R into batch
        q = x_query.permute(0,2,3,1).reshape(B*R, Dq, C)
        k = x_key.permute(0,2,3,1).reshape(B*R, Dk, C)
        v = k.clone()
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q = q.reshape(B*R, Dq, H, hd).permute(0,2,1,3)
        k = k.reshape(B*R, Dk, H, hd).permute(0,2,1,3)
        v = v.reshape(B*R, Dk, H, hd).permute(0,2,1,3)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1,2).reshape(B*R, Dq, C)
        out = self.drop(self.proj(out))
        return out.reshape(B, R, Dq, C).permute(0,3,1,2)  # (B,C,R,Dq)

class RACAFusionBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: int=4):
        super().__init__()
        self.raca_ra = RangeAnchoredCrossAttention(in_channels, num_heads)
        self.raca_re = RangeAnchoredCrossAttention(in_channels, num_heads)
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 1), nn.GELU(),
            nn.Conv2d(in_channels*2, in_channels, 1),
        )
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)

    def forward(self, x_rd, x_ra, x_re):
        fused = x_rd + self.raca_ra(x_rd, x_ra) + self.raca_re(x_rd, x_re)
        return self.ff(self.norm(fused)) + fused

class FeatureEmbedder(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        g = min(8, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, embed_dim, 3, padding=1), nn.GroupNorm(g, embed_dim), nn.ReLU(True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.GroupNorm(g, embed_dim), nn.ReLU(True),
        )
    def forward(self, x): return self.conv(x)

class RACA_Classifier(nn.Module):
    """
    Input:
        rd : (B, T, R, D)  R=128 D=64
        ra : (B, T, A, R)  A=361 R=128
        re : (B, T, A, R)
    """
    def __init__(self, num_classes: int, embed_dim: int=64,
                 lstm_hidden: int=128, lstm_layers: int=2,
                 num_heads: int=4, dropout: float=0.2,
                 angle_reduced: int=64):
        super().__init__()
        self.embed_rd = FeatureEmbedder(embed_dim)
        self.embed_ra = FeatureEmbedder(embed_dim)
        self.embed_re = FeatureEmbedder(embed_dim)
        # reduce A=361 -> angle_reduced before cross-attn (saves memory)
        self.ra_pool = nn.AdaptiveAvgPool2d((None, angle_reduced))
        self.re_pool = nn.AdaptiveAvgPool2d((None, angle_reduced))
        self.fusion = RACAFusionBlock(embed_dim, num_heads)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, num_classes)

    def forward(self, rd, ra, re):
        B, T, R, D = rd.shape
        A = ra.shape[2]
        # flatten time
        rd_in = rd.reshape(B*T, R, D).unsqueeze(1)            # (B*T,1,R,D)
        ra_in = ra.reshape(B*T, A, R).permute(0,2,1).unsqueeze(1)  # (B*T,1,R,A)
        re_in = re.reshape(B*T, A, R).permute(0,2,1).unsqueeze(1)
        # embed
        f_rd = self.embed_rd(rd_in)   # (B*T,C,R,D)
        f_ra = self.embed_ra(ra_in)   # (B*T,C,R,A)
        f_re = self.embed_re(re_in)
        # downsample angle
        f_ra = self.ra_pool(f_ra)     # (B*T,C,R,angle_reduced)
        f_re = self.re_pool(f_re)
        # fuse
        fused = self.fusion(f_rd, f_ra, f_re)   # (B*T,C,R,D)
        # spatial pool
        spatial = self.spatial_pool(fused).flatten(1)  # (B*T,C)
        # temporal
        temporal = spatial.view(B, T, -1)              # (B,T,C)
        lstm_out, _ = self.lstm(temporal)
        feat = self.drop(lstm_out[:, -1, :])
        return self.head(feat)


# ---------------------------------------------------------------------------
# Training / eval loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for rd, ra, re, labels in loader:
        rd, ra, re, labels = rd.to(device), ra.to(device), re.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(rd, ra, re)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for rd, ra, re, labels in loader:
        rd, ra, re, labels = rd.to(device), ra.to(device), re.to(device), labels.to(device)
        out = model(rd, ra, re)
        total_loss += criterion(out, labels).item()
        preds = out.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = 100.0 * correct / total if total else 0.0
    return total_loss / len(loader), acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train RACA on radar features.")
    p.add_argument("--manifest",      default="dataset/train_manifest.jsonl")
    p.add_argument("--out-dir",       default="checkpoints")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--val-split",     type=float, default=0.2)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--embed-dim",     type=int,   default=64)
    p.add_argument("--lstm-hidden",   type=int,   default=128)
    p.add_argument("--lstm-layers",   type=int,   default=2)
    p.add_argument("--angle-reduced", type=int,   default=64)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        default="")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[train] device={device}")

    # --- Dataset ---
    dataset = RadarManifestDataset(args.manifest)
    label_map   = dataset.label_map
    class_names = [dataset.idx_to_label[i] for i in range(len(label_map))]
    print(f"[train] classes: {class_names}")
    print(f"[train] total samples: {len(dataset)}")

    # --- Stratified split ---
    indices    = np.arange(len(dataset))
    labels_arr = np.array(dataset.labels)
    try:
        train_idx, val_idx = train_test_split(
            indices, test_size=args.val_split,
            stratify=labels_arr, random_state=args.seed
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices, test_size=args.val_split, random_state=args.seed
        )

    print(f"[train] train={len(train_idx)}  val={len(val_idx)}")
    print("[train] train dist:", {class_names[k]: v for k, v in Counter(labels_arr[train_idx].tolist()).items()})
    print("[train] val   dist:", {class_names[k]: v for k, v in Counter(labels_arr[val_idx].tolist()).items()})

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # --- Model ---
    model = RACA_Classifier(
        num_classes=len(class_names),
        embed_dim=args.embed_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        angle_reduced=args.angle_reduced,
    ).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer & scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Class weights to handle imbalance ---
    counts    = Counter(labels_arr[train_idx].tolist())
    weights   = torch.tensor(
        [1.0 / counts[i] for i in range(len(class_names))], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc  = 0.0
    best_path = os.path.join(args.out_dir, "raca_best.pth")

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_preds, va_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.1f}% | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.1f}% | "
              f"{elapsed:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "label_map":  label_map,
                "class_names": class_names,
                "val_acc":    va_acc,
                "args":       vars(args),
            }, best_path)
            print(f"  -> saved best model (val_acc={va_acc:.1f}%)")

    print(f"\n[train] done. best val acc={best_acc:.1f}%  model={best_path}")

    # --- Final evaluation on val set with best model ---
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)

    print("\n" + "="*60)
    print("Classification Report (validation set):")
    print(classification_report(final_labels, final_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(final_labels, final_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save label map for inference
    label_map_path = os.path.join(args.out_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label_map": label_map, "class_names": class_names}, f, indent=2)
    print(f"[train] label map saved to {label_map_path}")


if __name__ == "__main__":
    main()
