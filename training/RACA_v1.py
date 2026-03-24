import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from collections import Counter

# ===================================================================
# Part 1: RACA 核心模块 
# ===================================================================

class RangeAnchoredCrossAttention(nn.Module):
    """
    距离锚点跨视图注意力 (RACA) 模块
    
    原理:
    毫米波雷达的 Range (距离) 维度是 RD, RA, RE 三个视图的物理共享锚点。
    该模块通过将 Range 维度合并入 Batch 维度，强制注意力机制仅在
    相同的距离单元 (Range Bin) 内部发生，从而实现物理对齐并减少计算量。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Query, Key, Value 投影
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_query, x_key):
        """
        参数:
            x_query: 主视图特征 (例如 RD 图谱), Shape: (B, C, R, D_q)
                     R=Range, D_q=Doppler (or other dim)
            x_key:   辅助视图特征 (例如 RA 图谱), Shape: (B, C, R, D_k)
                     R=Range, D_k=Angle (or other dim)
        
        返回:
            out: 融合后的特征，保持 x_query 的形状 (B, C, R, D_q)
        """
        B, C, R, D_q = x_query.shape
        _, _, _, D_k = x_key.shape

        # ---------------------------------------------------------
        # 1. 物理锚定变换 (Range-Anchoring)
        # ---------------------------------------------------------
        # 我们需要将形状变换为 (Batch * Range, Sequence_Length, Channel)
        # 这样做的物理意义是：每个 Range Bin 被视为一个独立的样本，
        # 注意力不会跨越不同的距离单元。
        
        # (B, C, R, D) -> (B, R, D, C) -> (B*R, D, C)
        q = x_query.permute(0, 2, 3, 1).reshape(B * R, D_q, C)
        k = x_key.permute(0, 2, 3, 1).reshape(B * R, D_k, C)
        v = x_key.permute(0, 2, 3, 1).reshape(B * R, D_k, C)

        # ---------------------------------------------------------
        # 2. 跨视图注意力计算 (Cross-View Attention)
        # ---------------------------------------------------------
        # 这里的 Sequence Length 分别是 Doppler 维度和 Angle 维度
        # 我们在寻找：对于特定的距离 R，哪个角度 (Angle) 的特征与当前速度 (Doppler) 最相关？
        
        # Q projection
        q = self.to_q(q) # (B*R, D_q, C)
        k = self.to_k(k) # (B*R, D_k, C)
        v = self.to_v(v) # (B*R, D_k, C)

        # Reshape for Multi-head: (B*R, Heads, Seq, Head_Dim)
        q = q.reshape(B * R, D_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B * R, D_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B * R, D_k, self.num_heads, -1).permute(0, 2, 1, 3)

        # Attention Score: (B*R, Heads, D_q, D_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Aggregation: (B*R, Heads, D_q, Head_Dim)
        out = attn @ v
        
        # Restore shape: (B*R, D_q, C)
        out = out.transpose(1, 2).reshape(B * R, D_q, C)
        out = self.proj(out)
        out = self.dropout(out)

        # ---------------------------------------------------------
        # 3. 恢复物理维度
        # ---------------------------------------------------------
        # (B*R, D_q, C) -> (B, R, D_q, C) -> (B, C, R, D_q)
        out = out.reshape(B, R, D_q, C).permute(0, 3, 1, 2)
        
        # 残差连接 (Residual Connection) 通常在外部做，但在模块内做也可
        # 这里返回 delta，由外部加到原始 query 上
        return out


class RACAFusionBlock(nn.Module):
    """
    多视图融合模块
    
    功能:
    以 RD (Range-Doppler) 图谱为主干，融合 RA (Range-Azimuth) 和 RE (Range-Elevation) 的特征。
    这使得 RD 图谱中的每个像素不仅包含速度信息，还聚合了该距离处对应的水平和垂直空间信息。
    """
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        
        # 假设所有视图的特征通道数相同，如果不同可以在这里加 1x1 卷积调整
        self.raca_rd_ra = RangeAnchoredCrossAttention(dim=in_channels, num_heads=num_heads)
        self.raca_rd_re = RangeAnchoredCrossAttention(dim=in_channels, num_heads=num_heads)
        
        # 融合后的特征处理层 (MLP)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
        )
        self.norm = nn.GroupNorm(8, in_channels)

    def forward(self, x_rd, x_ra, x_re):
        """
        输入:
            x_rd: (Batch, C, Range, Doppler)
            x_ra: (Batch, C, Range, Azimuth)
            x_re: (Batch, C, Range, Elevation)
        
        注意: 三者的 Range 维度大小必须一致 (物理对齐的前提)。
        Doppler, Azimuth, Elevation 的维度大小可以不同 (RACA 允许解耦分辨率)。
        """
        assert x_rd.shape[2] == x_ra.shape[2] == x_re.shape[2], \
            "物理对齐错误: 所有视图的 Range 维度必须一致"

        # 1. 融合 RA -> RD
        # "在距离 R 处，这个速度特征对应哪个方位角?"
        ra_feat = self.raca_rd_ra(x_query=x_rd, x_key=x_ra)
        
        # 2. 融合 RE -> RD
        # "在距离 R 处，这个速度特征对应哪个高度?"
        re_feat = self.raca_rd_re(x_query=x_rd, x_key=x_re)

        # 3. 聚合特征
        # 将空间信息注入到 RD 特征中
        fused = x_rd + ra_feat + re_feat
        
        # 4. 前馈网络与归一化
        out = self.norm(fused)
        out = self.feed_forward(out) + out
        
        return out

# ===================================================================
# Part 2: RACA 分类器架构
# ===================================================================

class FeatureEmbedder(nn.Module):
    """
    特征嵌入层: 将原始的雷达图谱 (1通道) 映射到高维特征空间 (C通道)
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 使用 1x1 卷积或者 3x3 卷积提取初步特征
        # 保持 Range 和 Bin 维度不变
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (Batch*Time, 1, Range, Bins)
        return self.conv(x)

class RACA_Classifier(nn.Module):
    def __init__(self, num_classes=4, d_bins=64, a_bins=64, e_bins=64, embed_dim=64):
        """
        参数:
            num_classes: 分类类别数 (walk, sit, stand, fall)
            d_bins: 多普勒 FFT 点数 (由 Radar.cfg 的 numLoops 决定，您的是 64)
            a_bins: 方位角 FFT 点数 (由 mmWave_RA_RE.py 的 ANGLE_FFT_SIZE 决定，通常为 64)
            e_bins: 俯仰角 FFT 点数 (同上)
            embed_dim: RACA 处理的特征维度
        """
        super().__init__()
        self.d_bins = d_bins
        self.a_bins = a_bins
        self.e_bins = e_bins
        
        # 1. 特征提取 (将原始信号映射到 embed_dim)
        # 我们对 RD, RA, RE 分别使用共享或独立的 Embedder
        # 这里为了保持物理特性的差异，建议使用独立 Embedder
        self.embed_rd = FeatureEmbedder(1, embed_dim)
        self.embed_ra = FeatureEmbedder(1, embed_dim)
        self.embed_re = FeatureEmbedder(1, embed_dim)
        
        # 2. RACA 融合模块 (直接使用内部类)
        self.raca_fusion = RACAFusionBlock(in_channels=embed_dim)
        
        # 3. 特征聚合 (融合后降低维度)
        # 将 (Range, Doppler) 维度的特征聚合为一个向量
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 时序分类头 (处理 Frames 维度)
        # 输入维度: embed_dim
        self.temporal_head = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        输入 x 来自 DataLoader: [Batch, Total_Bins, Frames, Range]
        Total_Bins = Doppler + Azimuth + Elevation
        """
        B, C_total, T, R = x.shape
        
        # -----------------------------------------------------------
        # Step 1: 数据解包 (Unpack)
        # -----------------------------------------------------------
        # 根据预处理时的拼接顺序: RD -> RA -> RE
        # x 的第 1 维是 bins 拼接维度。
        # 我们需要先转置，把 Bins 放到最后，方便切分，或者直接在维度 1 切分。
        
        # 校验维度
        expected_bins = self.d_bins + self.a_bins + self.e_bins
        assert C_total == expected_bins, \
            f"Input channels {C_total} does not match sum of bins {expected_bins}. Check config."

        # 切分
        # x_rd_raw: [B, D_bins, T, R]
        x_rd_raw = x[:, :self.d_bins, :, :]
        # x_ra_raw: [B, A_bins, T, R]
        x_ra_raw = x[:, self.d_bins : self.d_bins+self.a_bins, :, :]
        # x_re_raw: [B, E_bins, T, R]
        x_re_raw = x[:, self.d_bins+self.a_bins :, :, :]

        # -----------------------------------------------------------
        # Step 2: 维度重塑以适配 RACA (Batch*Time 融合)
        # -----------------------------------------------------------
        # 目标格式: [Batch*Time, 1, Range, Bins]
        # 原始: [B, Bins, T, R] -> [B, T, R, Bins] -> [B*T, 1, R, Bins]
        
        # Helper function for reshaping
        def prepare_view(tensor, bins):
            # [B, Bins, T, R] -> [B, T, 1, R, Bins]
            t = tensor.permute(0, 2, 3, 1).unsqueeze(2)
            # Merge B and T -> [B*T, 1, R, Bins]
            return t.reshape(B * T, 1, R, bins)

        rd_in = prepare_view(x_rd_raw, self.d_bins)
        ra_in = prepare_view(x_ra_raw, self.a_bins)
        re_in = prepare_view(x_re_raw, self.e_bins)

        # -----------------------------------------------------------
        # Step 3: 特征嵌入 (Embedding)
        # -----------------------------------------------------------
        # Input: [B*T, 1, R, Bin] -> Output: [B*T, Embed_Dim, R, Bin]
        feat_rd = self.embed_rd(rd_in)
        feat_ra = self.embed_ra(ra_in)
        feat_re = self.embed_re(re_in)

        # -----------------------------------------------------------
        # Step 4: RACA 融合
        # -----------------------------------------------------------
        # 融合后的 RD 特征: [B*T, Embed_Dim, R, D_bins]
        fused_rd = self.raca_fusion(feat_rd, feat_ra, feat_re)

        # -----------------------------------------------------------
        # Step 5: 时序分类
        # -----------------------------------------------------------
        # 5.1 空间池化: [B*T, C, R, D] -> [B*T, C, 1, 1] -> [B*T, C]
        spatial_feat = self.pool(fused_rd).flatten(1)
        
        # 5.2 恢复时序维度: [B, T, C]
        temporal_input = spatial_feat.view(B, T, -1)
        
        # 5.3 LSTM / Transformer
        # out: [B, T, Hidden], hn: [Layers, B, Hidden]
        # 我们取最后一个时间步的输出用于分类
        lstm_out, _ = self.temporal_head(temporal_input)
        final_feat = lstm_out[:, -1, :] # Take last frame feature
        
        logits = self.classifier(final_feat)
        
        return logits

# ===================================================================
# 训练代码示例
# ===================================================================
if __name__ == "__main__":
    from radar_dataset import RadarHeatmapDataset
    
    # 1. 配置参数
    CFG = {
        "d_bins": 64,   # Doppler Bins
        "a_bins": 64,   # Azimuth Bins
        "e_bins": 64,   # Elevation Bins
        "batch_size": 4, # 您的数据较少，保持小 batch size
        "lr": 1e-4,     # 学习率
        "epochs": 30,   # 训练轮数
        "data_dir": r"/home/zhang/Yuhong_Wu/Heatmap_Features", # 您的实际路径
        "save_dir": "./checkpoints_v1" # 模型保存路径
    }
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备数据
    if not os.path.exists(CFG["data_dir"]):
        print(f"Error: Data directory not found: {CFG['data_dir']}")
        exit(1)

    full_dataset = RadarHeatmapDataset(CFG["data_dir"])
    class_names = getattr(full_dataset, 'classes', ["walk", "sit", "stand", "fall"]) # 获取类别名
    
    # =======================================================================
    # 使用 sklearn 进行分层划分 (Stratified Split)
    # =======================================================================
    # 获取所有标签
    all_labels = full_dataset.labels # 假设 dataset.labels 是一个列表
    all_indices = np.arange(len(full_dataset))
    
    try:
        # stratify 参数确保了划分后的标签比例与原数据一致
        train_idx, val_idx = train_test_split(
            all_indices, 
            test_size=0.2, 
            stratify=all_labels, 
            random_state=42 # 固定随机种子，保证每次划分结果一致
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed (probably a class has only 1 sample). Falling back to random split. Error: {e}")
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

    # 使用 Subset 创建数据集
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # 打印分布情况
    def print_distribution(dataset_indices, labels, name):
        subset_labels = [labels[i] for i in dataset_indices]
        counts = Counter(subset_labels)
        print(f"\n--- {name} Set Distribution ---")
        for idx, count in counts.items():
            cls_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
            print(f"  {cls_name}: {count}")
    
    print_distribution(train_idx, all_labels, "Training")
    print_distribution(val_idx, all_labels, "Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"], shuffle=False)
    
    print(f"\nTotal: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 3. 初始化模型
    model = RACA_Classifier(
        num_classes=4, 
        d_bins=CFG["d_bins"], 
        a_bins=CFG["a_bins"], 
        e_bins=CFG["e_bins"]
    ).to(device)
    
    # 4. 优化器与损失
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 确保保存目录存在
    os.makedirs(CFG["save_dir"], exist_ok=True)
    
    # 5. 训练循环
    print("\nStart Training...")
    best_acc = 0.0
    
    for epoch in range(CFG["epochs"]):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch+1}/{CFG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CFG["save_dir"], "raca_best.pth")
            torch.save(model.state_dict(), save_path)
            # print(f"  -> Model saved to {save_path}")

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.2f}%")
    
    # ===================================================================
    # 6. 生成混淆矩阵与分类报告
    # ===================================================================
    print("\nGenerating Confusion Matrix for the Best Model...")
    
    best_model_path = os.path.join(CFG["save_dir"], "raca_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model weights for evaluation.")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    
    print("\nClassification Report:")
    # 使用 zero_division=0 避免除零警告
    print(classification_report(all_targets, all_preds, labels=range(len(class_names)), target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(cm)
    
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - RACA Net')
        
        cm_save_path = os.path.join(CFG["save_dir"], "confusion_matrix.png")
        plt.savefig(cm_save_path)
        print(f"Confusion matrix plot saved to {cm_save_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Plotting failed: {e}")