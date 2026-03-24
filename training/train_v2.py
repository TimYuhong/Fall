"""RACA v2 -- 适配本项目特征格式的训练脚本

本项目特征 (由 offline/feature_extractor.py 生成):
    RD.npy : (T=100, range=128, doppler=64)   -- float32
    RA.npy : (T=100, angle=361, range=128)    -- float32
    RE.npy : (T=100, angle=361, range=128)    -- float32

优化说明 (v2.1):
    1. RE 分支加权 (--re-weight 0.3):
       IWR6843ISK 仅有 2 个俯仰虚拟阵元，仰角分辨率约 58°，RE 携带的
       独立信息量远少于 RA。通过 re_weight < 1.0 降低 RE 分支的融合权重，
       避免低质量 RE 引入额外噪声干扰 RD 主干。
    2. 二分类模式 (--binary):
       跌倒检测本质是二分类：fall vs non-fall。开启后使用
       BCEWithLogitsLoss，正类权重由 pos_weight 自动计算，
       对不均衡数据更鲁棒，且输出单 logit 更简洁。
    3. 时序模型选择建议:
       LSTM  —— 推荐用于跌倒检测（数据集小、动作时序显著、顺序依赖强）
       Transformer -- 适合大数据集（>1000样本）或需要捕捉长程依赖时

用法::
    # 二分类跌倒检测（推荐）
    python -m training.train_v2 \\
        --manifest dataset/train_manifest.jsonl \\
        --out-dir checkpoints \\
        --binary --force-model lstm --re-weight 0.3

    # 多分类动作识别
    python -m training.train_v2 \\
        --manifest dataset/train_manifest.jsonl \\
        --out-dir checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===========================================================================
# 1. Dataset
# ===========================================================================

class RadarManifestDataset(Dataset):
    """从 train_manifest.jsonl 加载 RD/RA/RE 特征。

    返回 (rd, ra, re, label):
        rd  : float32 (T, R, D)   T=100, R=128, D=64
        ra  : float32 (T, A, R)   A=361, R=128
        re  : float32 (T, A, R)
        label : int64
    """

    def __init__(self, manifest_path: str,
                 label_map: Optional[Dict[str, int]] = None):
        self.records: List[Dict] = []
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        all_labels = sorted({r["label"] for r in self.records})
        self.label_map: Dict[str, int] = label_map or {l: i for i, l in enumerate(all_labels)}
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.labels = [self.label_map[r["label"]] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        
        # 兼容跨平台路径：如果 manifest 存的是 Windows 绝对路径，
        # 在 Linux 服务器上运行时提取相对路径 (从 'dataset/' 开始)
        def resolve_path(p: str) -> str:
            p = p.replace("\\", "/")
            if "dataset/" in p:
                return os.path.join("dataset", p.split("dataset/")[-1])
            return p

        rd = np.load(resolve_path(rec["rd_path"])).astype(np.float32)   # (T, R, D)
        ra = np.load(resolve_path(rec["ra_path"])).astype(np.float32)   # (T, A, R)
        re = np.load(resolve_path(rec["re_path"])).astype(np.float32)   # (T, A, R)

        return (
            torch.from_numpy(rd),
            torch.from_numpy(ra),
            torch.from_numpy(re),
            torch.tensor(self.label_map[rec["label"]], dtype=torch.long),
        )


def collate_fn(batch):
    rds, ras, res, labels = zip(*batch)
    return torch.stack(rds), torch.stack(ras), torch.stack(res), torch.stack(labels)


# ===========================================================================
# 2. 模型: RACA (汲取自 RACA_v1.py + train_raca.py)
# ===========================================================================

class RangeAnchoredCrossAttention(nn.Module):
    """距离锚点跨视图注意力 (来自 RACA_v1.py).

    Range 维度合并进 Batch，强制注意力仅在同一距离单元内发生，
    实现物理对齐并降低计算量。
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query: torch.Tensor, x_key: torch.Tensor) -> torch.Tensor:
        """x_query/x_key: (B, C, R, Dq/Dk) -> out: (B, C, R, Dq)"""
        B, C, R, Dq = x_query.shape
        Dk = x_key.shape[3]
        H, hd = self.num_heads, C // self.num_heads

        # 物理锚定: 合并 B*R
        q = x_query.permute(0, 2, 3, 1).reshape(B * R, Dq, C)
        k = x_key.permute(0, 2, 3, 1).reshape(B * R, Dk, C)
        v = k.clone()

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)

        q = q.reshape(B*R, Dq, H, hd).permute(0, 2, 1, 3)
        k = k.reshape(B*R, Dk, H, hd).permute(0, 2, 1, 3)
        v = v.reshape(B*R, Dk, H, hd).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B * R, Dq, C)
        out = self.dropout(self.proj(out))
        return out.reshape(B, R, Dq, C).permute(0, 3, 1, 2)  # (B,C,R,Dq)


class RACAFusionBlock(nn.Module):
    """多视图融合块 (来自 RACA_v1.py + train_raca.py).

    以 RD 为主干，融合 RA 和 RE 的空间信息。
    re_weight < 1.0 可降低低质量 RE 分支（ISK 仅 2 俯仰阵元）的噪声影响。
    """
    def __init__(self, in_channels: int, num_heads: int = 4, dropout: float = 0.1,
                 re_weight: float = 1.0):
        super().__init__()
        self.re_weight = re_weight
        self.raca_ra = RangeAnchoredCrossAttention(in_channels, num_heads, dropout)
        self.raca_re = RangeAnchoredCrossAttention(in_channels, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels * 2, in_channels, 1),
        )
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)

    def forward(self, x_rd, x_ra, x_re):
        assert x_rd.shape[2] == x_ra.shape[2] == x_re.shape[2], \
            "物理对齐错误: Range 维度必须一致"
        # RE 分支加权：对于 IWR6843ISK 建议 re_weight=0.3 降低低分辨率 RE 干扰
        fused = x_rd + self.raca_ra(x_rd, x_ra) + self.re_weight * self.raca_re(x_rd, x_re)
        return self.ff(self.norm(fused)) + fused


class FeatureEmbedder(nn.Module):
    """单通道 2D 图谱 -> embed_dim 特征 (来自 RACA_v1.py)."""
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        g = min(8, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(1, embed_dim, 3, padding=1),
            nn.GroupNorm(g, embed_dim), nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(g, embed_dim), nn.ReLU(True),
        )
    def forward(self, x): return self.conv(x)


class TemporalLSTM(nn.Module):
    """LSTM 时序建模 -- 适合小数据集 (来自 train_raca.py)."""
    def __init__(self, dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = hidden_dim

    def forward(self, x):          # x: (B, T, D)
        out, _ = self.lstm(x)
        return out[:, -1, :]       # 最后时间步


class TemporalTransformer(nn.Module):
    """Transformer 时序建模 -- 适合较大数据集 (来自 train_raca.py)."""
    def __init__(self, dim: int, depth: int = 3, heads: int = 4,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x):          # x: (B, T, D)
        b, t, _ = x.shape
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos_emb[:, :t + 1])
        return self.norm(self.encoder(x))[:, 0]   # CLS token


class RACA_v2(nn.Module):
    """完整 RACA 分类器，适配本项目特征格式。

    输入:
        rd : (B, T, R, D)   R=128 D=64
        ra : (B, T, A, R)   A=361 R=128
        re : (B, T, A, R)

    参数:
        binary_mode : True → 二分类（fall vs non-fall），输出单 logit，
                      搭配 BCEWithLogitsLoss 使用；False → 多分类。
        re_weight   : RE 分支融合权重，ISK 硬件建议设为 0.3。
    """
    def __init__(self, num_classes: int, embed_dim: int = 64,
                 temporal_type: str = "lstm", temporal_cfg: Optional[Dict] = None,
                 dropout: float = 0.2, angle_reduced: int = 64,
                 binary_mode: bool = False, re_weight: float = 1.0):
        super().__init__()
        cfg = temporal_cfg or {}
        self.binary_mode = binary_mode

        self.embed_rd = FeatureEmbedder(embed_dim, dropout)
        self.embed_ra = FeatureEmbedder(embed_dim, dropout)
        self.embed_re = FeatureEmbedder(embed_dim, dropout)

        # 将 A=361 降至 angle_reduced 再做 RACA (省显存)
        self.ra_pool = nn.AdaptiveAvgPool2d((None, angle_reduced))
        self.re_pool = nn.AdaptiveAvgPool2d((None, angle_reduced))

        # RE 加权融合：ISK 俯仰分辨率约 58°，建议 re_weight=0.3
        self.fusion = RACAFusionBlock(embed_dim, num_heads=4, dropout=dropout,
                                     re_weight=re_weight)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        if temporal_type == "transformer":
            self.temporal = TemporalTransformer(
                dim=embed_dim,
                depth=cfg.get("depth", 3),
                heads=cfg.get("heads", 4),
                max_len=cfg.get("max_len", 128),
                dropout=dropout,
            )
        else:
            self.temporal = TemporalLSTM(
                dim=embed_dim,
                hidden_dim=cfg.get("hidden_dim", 128),
                num_layers=cfg.get("num_layers", 2),
                dropout=dropout,
            )

        # 分类头
        # 二分类: 输出 1 个 logit（配合 BCEWithLogitsLoss）
        # 多分类: 输出 num_classes 个 logit（配合 CrossEntropyLoss）
        out_dim = 1 if binary_mode else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal.out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def _spatial_forward(self, rd_b, ra_b, re_b):
        f_rd = self.embed_rd(rd_b)                  # (T,C,R,D)
        f_ra = self.embed_ra(self.ra_pool(ra_b))    # (T,C,R,A_small)
        f_re = self.embed_re(self.re_pool(re_b))    # (T,C,R,A_small)

        fused = self.fusion(f_rd, f_ra, f_re)       # (T,C,R,D)
        return self.spatial_pool(fused).flatten(1)  # (T,C)

    def forward(self, rd, ra, re):
        from torch.utils.checkpoint import checkpoint
        B, T, R, D = rd.shape          # rd: (B,T,R=128,D=64)
        A = ra.shape[2]                # ra: (B,T,A=361,R=128) -> A=361

        # 优化显存终极方案：将 B*T 切分为 T 逐批(batch)通过 CNN
        # 并在此基础上引入 PyTorch Checkpoint 机制 (梯度检查点)！
        # CNN + Attention 的中间激活缓存极其吃显存（一秒100帧，对2D CNN如同 batch=100 的图像）。
        # 用 checkpoint 丢弃中间缓存，反向传播时再临时重算，显存瞬间从 24G 降低到几个 G！
        spatial_feats = []
        for b in range(B):
            rd_b = rd[b].unsqueeze(1)                   # (T,1,R=128,D=64)
            ra_b = ra[b].permute(0, 2, 1).unsqueeze(1)  # (T,1,R=128,A=361)
            re_b = re[b].permute(0, 2, 1).unsqueeze(1)  # (T,1,R=128,A=361)

            if self.training:
                # 必须指定 use_reentrant=False 适应现代 PyTorch 规范
                spatial = checkpoint(self._spatial_forward, rd_b, ra_b, re_b, use_reentrant=False)
            else:
                spatial = self._spatial_forward(rd_b, ra_b, re_b)
                
            spatial_feats.append(spatial)

        temporal_in = torch.stack(spatial_feats, dim=0) # (B,T,C)
        feat = self.temporal(temporal_in)               # (B, out_dim)
        return self.classifier(feat)                           # (B,1) or (B,num_classes)


# ===========================================================================
# 3. 自适应配置 (汲取自 train_raca.py get_adaptive_config)
# ===========================================================================

def get_adaptive_config(num_samples: int, force_model: Optional[str] = None,
                        binary_mode: bool = False) -> Dict:
    """根据数据量自动生成最优训练配置。

    跌倒检测（binary_mode=True）场景说明：
    - LSTM 更适合：跌倒动作有明显的时序因果关系（站立→跌倒→卧地），
      LSTM 天然建模顺序依赖，参数少，小数据集不易过拟合。
    - Transformer 适合：数据集大（>1000样本）且需要捕捉帧间长程关联时。
    - 二分类时关闭 label_smoothing（边界需要清晰）。
    """
    if num_samples < 200:
        cfg = dict(temporal_type="lstm", batch_size=4, epochs=80,
                   lr=1e-4, weight_decay=1e-3, dropout=0.3,
                   label_smoothing=0.0 if binary_mode else 0.1, val_split=0.2,
                   temporal_cfg=dict(hidden_dim=128, num_layers=2))
    elif num_samples < 500:
        cfg = dict(temporal_type="lstm", batch_size=8, epochs=100,
                   lr=2e-4, weight_decay=5e-4, dropout=0.25,
                   label_smoothing=0.0 if binary_mode else 0.05, val_split=0.2,
                   temporal_cfg=dict(hidden_dim=128, num_layers=2))
    elif num_samples < 1200:
        # 二分类小数据集仍优先 LSTM；多分类可 Transformer
        temporal = "lstm" if binary_mode else "transformer"
        cfg = dict(temporal_type=temporal, batch_size=16, epochs=100,
                   lr=3e-4, weight_decay=1e-4, dropout=0.2,
                   label_smoothing=0.0, val_split=0.15,
                   temporal_cfg=dict(hidden_dim=128, num_layers=2) if binary_mode
                               else dict(depth=3, heads=4, max_len=128))
    else:
        temporal = "lstm" if binary_mode else "transformer"
        cfg = dict(temporal_type=temporal, batch_size=32, epochs=150,
                   lr=5e-4, weight_decay=1e-4, dropout=0.1,
                   label_smoothing=0.0, val_split=0.1,
                   temporal_cfg=dict(hidden_dim=256, num_layers=3) if binary_mode
                               else dict(depth=4, heads=8, max_len=128))
    if force_model == "lstm":
        cfg["temporal_type"] = "lstm"
        cfg.setdefault("temporal_cfg", {}).update(dict(hidden_dim=128, num_layers=2))
    elif force_model == "transformer":
        cfg["temporal_type"] = "transformer"
        cfg.setdefault("temporal_cfg", {}).update(dict(depth=3, heads=4, max_len=128))
    return cfg


# ===========================================================================
# 4. 训练 / 验证循环
# ===========================================================================

def train_epoch(model, loader, criterion, optimizer, device, binary_mode: bool = False):
    model.train()
    total_loss = correct = total = 0
    for rd, ra, re, labels in loader:
        rd, ra, re, labels = rd.to(device), ra.to(device), re.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(rd, ra, re)   # (B,1) or (B,num_classes)
        if binary_mode:
            loss = criterion(out.squeeze(1), labels.float())
            preds = (out.squeeze(1).sigmoid() >= 0.5).long()
        else:
            loss = criterion(out, labels)
            preds = out.argmax(1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, binary_mode: bool = False):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for rd, ra, re, labels in loader:
        rd, ra, re, labels = rd.to(device), ra.to(device), re.to(device), labels.to(device)
        out = model(rd, ra, re)
        if binary_mode:
            total_loss += criterion(out.squeeze(1), labels.float()).item()
            preds = (out.squeeze(1).sigmoid() >= 0.5).long()
        else:
            total_loss += criterion(out, labels).item()
            preds = out.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = 100.0 * correct / total if total else 0.0
    return total_loss / len(loader), acc, all_preds, all_labels


def save_plots(history: Dict, class_names: List, all_preds, all_labels, save_dir: str):
    """保存训练曲线 + 混淆矩阵 (来自 train_raca.py)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"],  label="Train")
    axes[1].plot(history["val_acc"],    label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, cmap="Blues")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix - RACA v2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    return cm


# ===========================================================================
# 5. Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RACA v2 训练脚本")
    p.add_argument("--manifest",     default="dataset/train_manifest.jsonl")
    p.add_argument("--out-dir",      default="checkpoints")
    p.add_argument("--force-model",  choices=["lstm", "transformer"], default=None,
                   help="强制指定时序模型类型 (默认根据数据量自动选择；二分类场景推荐 lstm)")
    p.add_argument("--embed-dim",    type=int,   default=64)
    p.add_argument("--angle-reduced",type=int,   default=64,
                   help="将 A=361 降采样到此值再做 RACA (省显存)")
    p.add_argument("--epochs",       type=int,   default=0,
                   help="覆盖自适应 epochs (0=自动)")
    p.add_argument("--batch-size",   type=int,   default=0,
                   help="覆盖自适应 batch_size (0=自动)")
    p.add_argument("--lr",           type=float, default=0.0,
                   help="覆盖自适应学习率 (0=自动)")
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="")
    p.add_argument("--binary",        action="store_true",
                   help="开启二分类模式 (fall vs non-fall)。需要 label_map 中含 'fall' 键")
    p.add_argument("--positive-label", default="fall",
                   help="二分类时正类标签名 (默认 'fall')")
    p.add_argument("--re-weight",     type=float, default=0.3,
                   help="RE 分支融合权重 (ISK 建议 0.3，AOP/ODS 可用 1.0)")
    p.add_argument("--patience",      type=int,   default=15,
                   help="Early Stopping 忍耐轮次 (0=关闭, 默认 15)")
    p.add_argument("--save-last",     action="store_true", default=True,
                   help="每 epoch 保存 last.pth 用于断点续训 (默认开启)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    binary_mode    = args.binary
    positive_label = args.positive_label
    re_weight      = args.re_weight

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"{'='*60}")
    print(f"RACA v2 训练脚本")
    print(f"{'='*60}")
    print(f"device      : {device}")
    print(f"manifest    : {args.manifest}")
    print(f"binary_mode : {binary_mode}")
    print(f"re_weight   : {re_weight}  (ISK 建议 0.3)")

    # ---- 全量数据集 (仅用于统计标签分布) ----
    full_ds = RadarManifestDataset(args.manifest)
    class_names = [full_ds.idx_to_label[i] for i in range(len(full_ds.label_map))]
    label_map   = full_ds.label_map
    print(f"classes     : {class_names}")
    print(f"total       : {len(full_ds)} samples")

    if binary_mode:
        if positive_label not in label_map:
            raise ValueError(
                f"--positive-label '{positive_label}' 不在 label_map 中。"
                f"可用标签: {list(label_map.keys())}"
            )
        fall_label = label_map[positive_label]
        # 重新映射标签：positive_label → 1，其余 → 0
        binary_label_map = {k: (1 if v == fall_label else 0) for k, v in label_map.items()}
        print(f"binary_label_map: {binary_label_map}")

    # ---- 自适应配置 ----
    cfg = get_adaptive_config(len(full_ds), force_model=args.force_model,
                               binary_mode=binary_mode)
    # 命令行覆盖
    if args.epochs    > 0: cfg["epochs"]     = args.epochs
    if args.batch_size > 0: cfg["batch_size"] = args.batch_size
    if args.lr        > 0: cfg["lr"]         = args.lr

    print(f"\n--- 自适应配置 (基于 {len(full_ds)} 样本) ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # ---- 数据集加载 ----
    effective_label_map = binary_label_map if binary_mode else label_map
    full_ds2 = RadarManifestDataset(args.manifest, effective_label_map)
    indices    = np.arange(len(full_ds2))
    labels_arr = np.array(full_ds2.labels)

    # ---- 8:1:1 三路分层划分 (train / val / test) ----
    try:
        train_idx, tmp_idx = train_test_split(
            indices, test_size=0.2,
            stratify=labels_arr, random_state=args.seed
        )
        tmp_labels = labels_arr[tmp_idx]
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.5,
            stratify=tmp_labels, random_state=args.seed
        )
    except ValueError:
        train_idx, tmp_idx = train_test_split(
            indices, test_size=0.2, random_state=args.seed
        )
        val_idx, test_idx = train_test_split(
            tmp_idx, test_size=0.5, random_state=args.seed
        )

    train_ds = RadarManifestDataset(args.manifest, effective_label_map)
    val_ds   = RadarManifestDataset(args.manifest, effective_label_map)
    test_ds  = RadarManifestDataset(args.manifest, effective_label_map)

    disp_names = (["non-fall", "fall"] if binary_mode
                  else [full_ds.idx_to_label[i] for i in range(len(label_map))])
    print(f"\ntrain={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    print("train dist:", {disp_names[k]: v for k, v in Counter(labels_arr[train_idx].tolist()).items()})
    print("val   dist:", {disp_names[k]: v for k, v in Counter(labels_arr[val_idx].tolist()).items()})
    print("test  dist:", {disp_names[k]: v for k, v in Counter(labels_arr[test_idx].tolist()).items()})

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds,  test_idx),  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)

    # ---- 模型 ----
    num_model_classes = 2 if binary_mode else len(class_names)
    model = RACA_v2(
        num_classes=num_model_classes,
        embed_dim=args.embed_dim,
        temporal_type=cfg["temporal_type"],
        temporal_cfg=cfg["temporal_cfg"],
        dropout=cfg["dropout"],
        angle_reduced=args.angle_reduced,
        binary_mode=binary_mode,
        re_weight=re_weight,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nmodel   : RACA_v2 (temporal={cfg['temporal_type'].upper()}, "
          f"binary={binary_mode}, re_weight={re_weight})")
    print(f"params  : {params:,}")

    # ---- 优化器 / 调度器 / 损失 ----
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)

    counts = Counter(labels_arr[train_idx].tolist())
    if binary_mode:
        # BCEWithLogitsLoss + pos_weight：正类不足时自动放大正类梯度
        n_neg = counts.get(0, 1)
        n_pos = counts.get(1, 1)
        pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        print(f"损失函数: BCEWithLogitsLoss  pos_weight={pos_w.item():.2f}")
    else:
        weights = torch.tensor([1.0 / counts[i] for i in range(len(class_names))],
                               dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights,
                                        label_smoothing=cfg["label_smoothing"])
        print(f"损失函数: CrossEntropyLoss  label_smoothing={cfg['label_smoothing']}")

    # ---- 输出目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(args.out_dir, f"{cfg['temporal_type']}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "raca_v2_best.pth")

    # ---- 训练循环 ----
    print(f"\n{'='*60}")
    print(f"开始训练 (epochs={cfg['epochs']}, batch={cfg['batch_size']}, lr={cfg['lr']})")
    print(f"{'='*60}")

    best_acc   = 0.0
    best_f1    = 0.0
    best_score = 0.0
    no_improve = 0          # Early Stopping 计数器
    patience   = args.patience
    last_path  = os.path.join(save_dir, "raca_v2_last.pth")  # 断点续训
    history: Dict = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                     "val_f1": [], "fall_recall": [], "val_score": []}

    if patience > 0:
        print(f"Early Stopping: patience={patience} epochs")
    # 二分类正类索引（0=non-fall, 1=fall）
    fall_idx_in_eval = 1 if binary_mode else effective_label_map.get(positive_label, 0)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                      binary_mode=binary_mode)
        va_loss, va_acc, va_preds, va_labels = eval_epoch(model, val_loader, criterion, device,
                                                           binary_mode=binary_mode)
        scheduler.step()

        # 综合评分: 宏F1(0.4) + 正类(fall)召回率(0.4) + 准确率(0.2)
        # 跌倒漏报比误报代价高，因此召回率权重最大
        va_f1      = f1_score(va_labels, va_preds, average="macro", zero_division=0)
        per_recall = recall_score(va_labels, va_preds, average=None, zero_division=0)
        fall_recall = float(per_recall[fall_idx_in_eval]) if fall_idx_in_eval < len(per_recall) else 0.0
        va_score = 0.4 * va_f1 + 0.4 * fall_recall + 0.2 * (va_acc / 100.0)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["fall_recall"].append(fall_recall)
        history["val_score"].append(va_score)

        lr_now = optimizer.param_groups[0]["lr"]
        marker = " *" if va_score > best_score else ""
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | "
              f"tr {tr_loss:.4f}/{tr_acc:.1f}% | "
              f"va {va_loss:.4f}/{va_acc:.1f}% f1={va_f1:.3f} fall_rec={fall_recall:.3f} score={va_score:.4f} | "
              f"lr={lr_now:.2e} | {time.time()-t0:.1f}s{marker}")

        # ---- Best model 保存 ----
        if va_score > best_score:
            best_score = va_score
            best_f1    = va_f1
            best_acc   = va_acc
            no_improve = 0
            torch.save({
                "epoch": epoch, "val_acc": va_acc, "val_f1": va_f1,
                "val_fall_recall": fall_recall, "val_score": va_score,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "label_map": effective_label_map,
                "class_names": disp_names,
                "binary_mode": binary_mode,
                "re_weight": re_weight,
                "cfg": cfg, "args": vars(args),
            }, best_path)
            print(f"  -> saved best (score={va_score:.4f}, f1={va_f1:.3f}, fall_rec={fall_recall:.3f}, acc={va_acc:.1f}%)")
        else:
            no_improve += 1

        # ---- Last checkpoint（每 epoch 覆盖，用于断点续训）----
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score, "no_improve": no_improve,
            "history": history,
            "label_map": effective_label_map,
            "class_names": disp_names,
            "binary_mode": binary_mode,
            "re_weight": re_weight,
            "cfg": cfg, "args": vars(args),
        }, last_path)

        # ---- Early Stopping ----
        if patience > 0 and no_improve >= patience:
            print(f"\nEarly Stopping: {patience} epochs without improvement. Stop at epoch {epoch}.")
            break

    print(f"\n训练完成. best score={best_score:.4f}  f1={best_f1:.3f}  acc={best_acc:.1f}%")
    print(f"  best_model : {best_path}")
    print(f"  last_model : {last_path}")

    # ---- 保存训练历史 JSON ----
    hist_path = os.path.join(save_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # ---- 最终 Val 评估 (best model) ----
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device,
                                                          binary_mode=binary_mode)

    print(f"\n{'='*60}")
    print("Classification Report (val, best model):")
    val_report = classification_report(final_labels, final_preds,
                                       target_names=disp_names, zero_division=0)
    print(val_report)

    n_disp = 2 if binary_mode else len(class_names)
    cm = save_plots(history, disp_names, final_preds, final_labels, save_dir)
    print("Confusion Matrix (val):")
    print(cm)

    # ---- Test 集评估（独立，不受 val best 影响）----
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device,
                                                       binary_mode=binary_mode)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {test_acc:.1f}%")
    print("Classification Report (test, best model):")
    test_report = classification_report(test_labels, test_preds,
                                        target_names=disp_names, zero_division=0)
    print(test_report)
    test_cm = confusion_matrix(test_labels, test_preds, labels=range(n_disp))
    print("Confusion Matrix (test):")
    print(test_cm)

    # ---- 保存文字报告至磁盘 ----
    report_path = os.path.join(save_dir, "test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"best model: {best_path}\n")
        f.write(f"best val score: {best_score:.4f}  f1: {best_f1:.3f}  acc: {best_acc:.1f}%\n")
        f.write(f"best epoch: {ckpt['epoch']}\n\n")
        f.write("=== Val Report (best model) ===\n")
        f.write(val_report + "\n")
        f.write(f"Val Confusion Matrix:\n{cm}\n\n")
        f.write(f"=== Test Report (best model) ===\n")
        f.write(f"Test Accuracy: {test_acc:.1f}%\n")
        f.write(test_report + "\n")
        f.write(f"Test Confusion Matrix:\n{test_cm}\n")
    print(f"\n报告已保存: {report_path}")
    print(f"历史已保存: {hist_path}")
    print(f"图表已保存: {save_dir}")

    # 保存标签映射
    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_map": effective_label_map, "class_names": disp_names,
                   "binary_mode": binary_mode}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
