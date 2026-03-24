"""
RACA 自适应训练脚本
===================
根据数据集规模自动调整模型架构和训练参数

使用方法:
    python train_raca.py --data_dir <path> [--force_model v1|v2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import datetime

# ===================================================================
# Part 1: 核心模块定义
# ===================================================================

class RangeAnchoredCrossAttention(nn.Module):
    """距离锚点跨视图注意力模块"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key):
        B, C, R, D_q = x_query.shape
        _, _, _, D_k = x_key.shape

        q = x_query.permute(0, 2, 3, 1).reshape(B * R, D_q, C)
        k = x_key.permute(0, 2, 3, 1).reshape(B * R, D_k, C)
        v = x_key.permute(0, 2, 3, 1).reshape(B * R, D_k, C)

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q = q.reshape(B * R, D_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B * R, D_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B * R, D_k, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B * R, D_q, C)
        out = self.proj(out)
        out = self.dropout(out)

        out = out.reshape(B, R, D_q, C).permute(0, 3, 1, 2)
        return out


class RACAFusionBlock(nn.Module):
    """多视图融合模块"""
    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.raca_rd_ra = RangeAnchoredCrossAttention(dim=in_channels, num_heads=num_heads, dropout=dropout)
        self.raca_rd_re = RangeAnchoredCrossAttention(dim=in_channels, num_heads=num_heads, dropout=dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        )
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x_rd, x_ra, x_re):
        assert x_rd.shape[2] == x_ra.shape[2] == x_re.shape[2], "物理对齐错误"
        ra_feat = self.raca_rd_ra(x_query=x_rd, x_key=x_ra)
        re_feat = self.raca_rd_re(x_query=x_rd, x_key=x_re)
        fused = x_rd + ra_feat + re_feat
        out = self.norm(fused)
        out = self.feed_forward(out) + out
        return out


class FeatureEmbedder(nn.Module):
    """特征嵌入层"""
    def __init__(self, in_channels=1, out_channels=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ===================================================================
# Part 2: 时序建模模块 (LSTM / Transformer)
# ===================================================================

class TemporalLSTM(nn.Module):
    """LSTM 时序建模 (适合小数据集)"""
    def __init__(self, dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)
        return out[:, -1, :]  # 最后时间步


class TemporalTransformer(nn.Module):
    """Transformer 时序建模 (适合大数据集)"""
    def __init__(self, dim, depth=3, heads=4, max_len=128, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x):
        b, f, d = x.shape
        
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        seq_len = x.shape[1]
        x += self.pos_emb[:, :seq_len]
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)
        x = self.norm(x)
        
        return x[:, 0]


# ===================================================================
# Part 3: 自适应 RACA 分类器
# ===================================================================

class AdaptiveRACA_Classifier(nn.Module):
    """
    自适应 RACA 分类器
    根据配置自动选择 LSTM 或 Transformer
    """
    def __init__(self, num_classes=4, d_bins=64, a_bins=64, e_bins=64, 
                 embed_dim=64, temporal_type='lstm', temporal_config=None, dropout=0.2):
        super().__init__()
        self.d_bins = d_bins
        self.a_bins = a_bins
        self.e_bins = e_bins
        
        self.embed_rd = FeatureEmbedder(1, embed_dim, dropout=dropout)
        self.embed_ra = FeatureEmbedder(1, embed_dim, dropout=dropout)
        self.embed_re = FeatureEmbedder(1, embed_dim, dropout=dropout)
        
        self.raca_fusion = RACAFusionBlock(in_channels=embed_dim, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 根据配置选择时序模型
        temporal_config = temporal_config or {}
        if temporal_type == 'transformer':
            self.temporal_head = TemporalTransformer(
                dim=embed_dim,
                depth=temporal_config.get('depth', 2),
                heads=temporal_config.get('heads', 4),
                max_len=temporal_config.get('max_len', 128),
                dropout=dropout
            )
        else:  # lstm
            self.temporal_head = TemporalLSTM(
                dim=embed_dim,
                hidden_dim=temporal_config.get('hidden_dim', 128),
                num_layers=temporal_config.get('num_layers', 2),
                dropout=dropout
            )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal_head.out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, C_total, T, R = x.shape
        
        x_rd_raw = x[:, :self.d_bins, :, :]
        x_ra_raw = x[:, self.d_bins:self.d_bins + self.a_bins, :, :]
        x_re_raw = x[:, self.d_bins + self.a_bins:, :, :]

        def prepare_view(tensor, bins):
            t = tensor.permute(0, 2, 3, 1).unsqueeze(2)
            return t.reshape(B * T, 1, R, bins)

        rd_in = prepare_view(x_rd_raw, self.d_bins)
        ra_in = prepare_view(x_ra_raw, self.a_bins)
        re_in = prepare_view(x_re_raw, self.e_bins)

        feat_rd = self.embed_rd(rd_in)
        feat_ra = self.embed_ra(ra_in)
        feat_re = self.embed_re(re_in)

        fused_rd = self.raca_fusion(feat_rd, feat_ra, feat_re)
        
        spatial_feat = self.pool(fused_rd).flatten(1)
        temporal_input = spatial_feat.view(B, T, -1)
        
        global_feat = self.temporal_head(temporal_input)
        logits = self.classifier(global_feat)
        
        return logits


# ===================================================================
# Part 4: 自适应配置生成器
# ===================================================================

def get_adaptive_config(num_samples, force_model=None):
    """
    根据数据量自动生成最优配置
    
    Args:
        num_samples: 数据集样本数
        force_model: 强制使用 'v1' (LSTM) 或 'v2' (Transformer)
    
    Returns:
        配置字典
    """
    config = {
        "d_bins": 64,
        "a_bins": 64,
        "e_bins": 64,
        "embed_dim": 64,
        "num_classes": 4,
    }
    
    # 根据数据量调整配置
    if num_samples < 100:
        # 小数据集: 强正则化, LSTM, 小 batch
        config.update({
            "temporal_type": "lstm" if force_model != 'v2' else "transformer",
            "batch_size": 4,
            "epochs": 50,
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "dropout": 0.3,
            "label_smoothing": 0.1,
            "test_size": 0.2,
            "temporal_config": {
                "hidden_dim": 128,
                "num_layers": 2,
                "depth": 2,
                "heads": 4,
            }
        })
        
    elif num_samples < 300:
        # 中等数据集
        config.update({
            "temporal_type": "lstm" if force_model == 'v1' else "transformer",
            "batch_size": 6,
            "epochs": 60,
            "lr": 3e-4,
            "weight_decay": 5e-4,
            "dropout": 0.25,
            "label_smoothing": 0.05,
            "test_size": 0.15,
            "temporal_config": {
                "hidden_dim": 128,
                "num_layers": 2,
                "depth": 3,
                "heads": 4,
            }
        })
        
    elif num_samples < 1000:
        # 较大数据集
        config.update({
            "temporal_type": "transformer",
            "batch_size": 16,
            "epochs": 100,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "dropout": 0.2,
            "label_smoothing": 0.05,
            "test_size": 0.15,
            "temporal_config": {
                "hidden_dim": 256,
                "num_layers": 3,
                "depth": 4,
                "heads": 8,
            }
        })
        
    else:
        # 大数据集
        config.update({
            "temporal_type": "transformer",
            "batch_size": 32,
            "epochs": 150,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "label_smoothing": 0.0,
            "test_size": 0.1,
            "temporal_config": {
                "hidden_dim": 256,
                "num_layers": 3,
                "depth": 6,
                "heads": 8,
            }
        })
    
    # 强制覆盖模型类型
    if force_model == 'v1':
        config["temporal_type"] = "lstm"
    elif force_model == 'v2':
        config["temporal_type"] = "transformer"
    
    return config


# ===================================================================
# Part 5: 训练与评估函数
# ===================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    # 如果是 DataParallel 模型，使用原始模块进行评估
    eval_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = eval_model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy, all_preds, all_targets


def save_confusion_matrix(targets, preds, class_names, save_path):
    """保存混淆矩阵图"""
    cm = confusion_matrix(targets, preds, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - RACA')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cm


# ===================================================================
# Part 6: 主程序
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='RACA 自适应训练脚本')
    parser.add_argument('--data_dir', type=str, default=r'/home/zhang/Yuhong_Wu/Heatmap_Features_25',
                        help='数据目录路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_adaptive_25',
                        help='模型保存路径')
    parser.add_argument('--force_model', type=str, choices=['v1', 'v2'], default=None,
                        help='强制使用 v1 (LSTM) 或 v2 (Transformer)')
    parser.add_argument('--re_scale', type=float, default=15.0,
                        help='RE通道放大倍数 (默认: 15.0)')
    args = parser.parse_args()
    
    # 导入改进的数据集（带Log压缩 + RE放大）
    import sys
    sys.path.append('/home/zhang/Yuhong_Wu')
    from radar_dataset_Pro import RadarDatasetWithLog
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"{'='*60}")
    print(f"RACA 自适应训练脚本")
    print(f"{'='*60}")
    print(f"Device: {device} (GPUs: {num_gpus})")
    print(f"Data Dir: {args.data_dir}")
    print(f"RE Scale: {args.re_scale}x")
    
    # 加载数据集
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    # 使用改进的数据集（解决17.7亿倍动态范围和RE通道过小问题）
    # 定义统计量文件路径
    stats_file_name = f'dataset_stats_log_{args.re_scale}x.npz'
    stats_file = os.path.join(os.path.dirname(args.data_dir), stats_file_name)
    
    # 第一步：创建full_dataset计算统计量
    print(f"\n[信息] 初始化数据集并计算统计量...")
    full_dataset = RadarDatasetWithLog(
        args.data_dir,
        use_log_compression=True,  # 关键：Log10压缩解决动态范围问题
        use_augmentation=True,    # 暂时不增强，后续在训练集单独设置
        re_scale_factor=args.re_scale,  # 使用命令行参数
        stats_file=stats_file      # 传入路径用于保存/加载
    )
    num_samples = len(full_dataset)
    
    if num_samples == 0:
        print("Error: Dataset is empty!")
        return
    
    class_names = getattr(full_dataset, 'classes', ["walk", "sit", "stand", "fall"])
    
    # 获取自适应配置
    config = get_adaptive_config(num_samples, force_model=args.force_model)
    
    print(f"\n{'='*60}")
    print(f"自适应配置 (基于 {num_samples} 个样本)")
    print(f"{'='*60}")
    print(f"  时序模型: {config['temporal_type'].upper()}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning Rate: {config['lr']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Label Smoothing: {config['label_smoothing']}")
    
    # 数据划分
    all_labels = full_dataset.labels
    all_indices = np.arange(num_samples)
    
    try:
        train_idx, val_idx = train_test_split(
            all_indices, 
            test_size=config['test_size'], 
            stratify=all_labels, 
            random_state=42
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed: {e}")
        train_idx, val_idx = train_test_split(all_indices, test_size=config['test_size'], random_state=42)

    # 第二步：创建训练集和验证集（复用统计量）
    print(f"\n[信息] 创建训练集和验证集（复用统计量）...")
    
    # 训练集：使用数据增强扩充样本
    train_dataset_full = RadarDatasetWithLog(
        args.data_dir,
        use_log_compression=True,
        use_augmentation=True,   # 训练时启用增强
        re_scale_factor=args.re_scale,  # 使用命令行参数
        stats_file=stats_file    # 复用统计量
    )
    train_dataset = Subset(train_dataset_full, train_idx)
    
    # 验证集：不使用数据增强
    val_dataset_full = RadarDatasetWithLog(
        args.data_dir,
        use_log_compression=True,
        use_augmentation=False,  # 验证时关闭增强
        re_scale_factor=args.re_scale,  # 使用命令行参数
        stats_file=stats_file    # 复用统计量
    )
    val_dataset = Subset(val_dataset_full, val_idx)
    
    # 打印分布
    print(f"\n--- 数据集分布 ---")
    for name, indices in [("Train", train_idx), ("Val", val_idx)]:
        subset_labels = [all_labels[i] for i in indices]
        counts = Counter(subset_labels)
        dist_str = ", ".join([f"{class_names[k]}: {v}" for k, v in sorted(counts.items())])
        print(f"  {name} ({len(indices)}): {dist_str}")
    
    # 创建 DataLoader (drop_last=True 避免多GPU时最后一个batch不完整)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True if num_gpus > 1 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    model = AdaptiveRACA_Classifier(
        num_classes=config['num_classes'],
        d_bins=config['d_bins'],
        a_bins=config['a_bins'],
        e_bins=config['e_bins'],
        embed_dim=config['embed_dim'],
        temporal_type=config['temporal_type'],
        temporal_config=config['temporal_config'],
        dropout=config['dropout']
    ).to(device)
    
    # 多GPU并行
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"\n[Multi-GPU] 使用 {num_gpus} 块 GPU 并行训练")
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {trainable_params:,} / {total_params:,}")
    
    # 优化器与损失函数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    # 创建保存目录 (模型类型/时间戳)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, config['temporal_type'], timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print(f"\n{'='*60}")
    print(f"开始训练...")
    print(f"{'='*60}")
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 每个 epoch 都显示完整训练细节
        lr = optimizer.param_groups[0]['lr']
        best_marker = " *" if val_acc > best_acc else ""
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {lr:.2e}{best_marker}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
    
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")
    
    # 最终评估
    print(f"\n{'='*60}")
    print(f"最终评估")
    print(f"{'='*60}")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, all_preds, all_targets = evaluate(model, val_loader, criterion, device)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, 
                                labels=range(len(class_names)), 
                                target_names=class_names, 
                                zero_division=0))
    
    # 保存混淆矩阵
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    cm = save_confusion_matrix(all_targets, all_preds, class_names, cm_path)
    print(f"\n混淆矩阵:")
    print(cm)
    print(f"\n混淆矩阵已保存至: {cm_path}")
    
    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"训练曲线已保存至: {curve_path}")
    
    print(f"\n所有结果保存在: {save_dir}")


if __name__ == "__main__":
    main()
