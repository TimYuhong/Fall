# IWR6843 实时跌倒检测与可视化系统 — 项目全景文档

> **版本日期**: 2026-03-16  
> **定位**: 基于 TI IWR6843 毫米波雷达的单人实时跌倒检测系统  
> **核心架构**: `Radar.cfg 配置 → UDP 采集 → DSP 信号处理 → 点云聚类 → 单目标跟踪 → 规则法跌倒检测 + 可选 ML 辅助`

---

## 目录

- [30 分钟阅读指南](#30-分钟阅读指南)
- [一、项目做了什么](#一项目做了什么)
- [二、整体数据链条](#二整体数据链条)
- [三、核心模块详解](#三核心模块详解)
- [四、界面与交互](#四界面与交互)
- [五、测试体系](#五测试体系)
- [六、依赖与环境](#六依赖与环境)
- [七、目录结构](#七目录结构)
- [八、模型接入指南](#八模型接入指南)
- [九、当前版本的边界与限制](#九当前版本的边界与限制)
- [十、后续推荐方向](#十后续推荐方向)

---

## 30 分钟阅读指南

如果你只有 30 分钟快速了解这个项目，按下面顺序阅读：

| 阶段 | 时间 | 读什么 | 目标 |
|------|------|--------|------|
| **1. 全局感知** | 5 min | 本文档的 [一、项目做了什么](#一项目做了什么) + [二、整体数据链条](#二整体数据链条) | 理解系统边界和数据流向 |
| **2. 配置与启动** | 3 min | `config/Radar.cfg` + `main.py` 前 100 行（import 和全局变量） | 理解运行时配置如何传递 |
| **3. 信号处理** | 5 min | `DSP.py` 的 `RDA_Time()` 和 `Range_Angle()` 函数签名 + 返回值 | 理解 5 张热图从哪来 |
| **4. 点云到目标** | 5 min | `pointcloud_clustering.py` 的 `cluster_pointcloud_simple()` | 理解聚类如何产生目标候选 |
| **5. 跟踪与检测** | 5 min | `target_tracking.py` 全文（104 行）+ `fall_detection.py` 的 `detect_fall()` | 理解跟踪和跌倒判定核心逻辑 |
| **6. ML 接口** | 3 min | `fall_predictor.py` 的 `BaseFallPredictor` / `NullFallPredictor` | 理解模型如何挂载 |
| **7. 主循环** | 4 min | `main.py` 的 `update_figure()` 函数中 `# 单目标稳定跟踪` 和 `# 高度检测` 注释段 | 理解一次刷新做了哪些事 |

> **提示**: 带着 "一帧雷达数据从进来到最终在界面上显示跌倒告警经历了什么？" 这个问题去读，效率最高。

---

## 一、项目做了什么

### 已实现的核心能力

| 能力 | 状态 | 关键模块 |
|------|------|----------|
| `Radar.cfg` 运行时配置解析 | ✅ 完整 | `iwr6843_tlv/detected_points.py`, `main.py` |
| 实时 UDP 数据采集与帧重组 | ✅ 完整 | `real_time_process.py` |
| DSP 信号处理（RTI/DTI/RDI/RAI/REI） | ✅ 完整 | `DSP.py` |
| 3D 点云提取与 DBSCAN 聚类 | ✅ 完整 | `DSP.py`, `pointcloud_clustering.py` |
| 单目标帧间稳定跟踪 | ✅ 完整 | `target_tracking.py` |
| 统一高度序列（显示与检测共用） | ✅ 完整 | `fall_detection.py`, `main.py` |
| 规则法跌倒检测（多特征融合，唯一策略） | ✅ 完整 | `fall_detection.py` |
| 跌倒状态面板实时驱动 | ✅ 完整 | `main.py` |
| 可选 Torch 模型辅助推理接口 | ✅ 完整 | `fall_predictor.py` |
| **离线特征提取器 (DSP 复用)** | ✅ 完整 | `offline_feature_extractor.py` |
| 多页签可视化 GUI | ✅ 完整 | `UI_interface.py`, `main.py` |
| `.bin` 回放接口 | 🔶 仅预留 | `replay_controller.py` |

### 设计原则

1. **单人优先** — 跟踪器按单目标设计，不保证多人场景
2. **规则法主判** — 正式报警由规则法触发，ML 结果仅作辅助显示
3. **显示 = 检测** — 高度曲线和跌倒判定使用完全相同的 `height_history`
4. **失锁即重置** — 目标跟踪丢失时自动清空历史，杜绝误拼跌倒事件

---

## 二、整体数据链条

### 全局数据流

```
┌──────────────┐     串口 cfg      ┌─────────────┐
│  Radar.cfg   │ ─────────────────→│  IWR6843    │
│  (配置真源)   │                   │  雷达硬件    │
└──────────────┘                   └──────┬──────┘
                                          │ UDP 原始 ADC
                                          ▼
┌──────────────────────────────────────────────────────┐
│                  real_time_process.py                 │
│  ┌──────────────┐        ┌──────────────────────┐    │
│  │ UdpListener  │──Raw──→│ DataProcessor        │    │
│  │ (采集线程)    │  Data  │ (处理线程)            │    │
│  └──────────────┘        └──────────┬───────────┘    │
└─────────────────────────────────────┼────────────────┘
                                      │ 调用 DSP
                                      ▼
┌──────────────────────────────────────────────────────┐
│                      DSP.py                          │
│                                                      │
│  RDA_Time() → RTI, DTI, RDI                          │
│  Range_Angle() → RAI, REI                            │
│  extract_pointcloud_from_angle_maps() → 3D 点云       │
│                                                      │
│  产出 → 6 个 Queue:                                   │
│  RTIData, DTIData, RDIData, RAIData, REIData,        │
│  PointCloudData                                      │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              main.py  update_figure()                │
│                                                      │
│  ① 刷新 5 张热图 (RTI/DTI/RDI/RAI/REI)               │
│  ② 合并点云历史 → 时间衰减显示                          │
│  ③ DBSCAN 聚类 → 候选目标中心列表                      │
│  ④ target_tracking.update_tracker() → 锁定主目标      │
│  ⑤ 主目标高度 → fd.update_height_history()            │
│  ⑥ fd.detect_fall() → 规则法跌倒判定                  │
│  ⑦ fall_predictor.predict() → ML 辅助推理 (200ms 节流) │
│  ⑧ 状态面板更新 (Standby/Tracking/No Target/          │
│     Fall Alert/Sample Saved)                         │
│                                                      │
│  刷新周期: QTimer.singleShot(1ms)                     │
└──────────────────────────────────────────────────────┘
```

### 跌倒检测子链（步骤 ③→⑥ 展开）

```
点云帧 [N×4: range, x, y, z]
        │
        ▼
   DBSCAN 聚类 (eps=0.3, min_samples=2)
        │
        ▼
   按能量降序排列的候选中心列表
        │
        ▼
   ┌────────────────────────────────┐
   │  target_tracking.update_tracker│
   │  ─ 已锁定? 找距离最近候选       │
   │  ─ 距离 ≤ gate_m? → tracked    │
   │  ─ 超限? → lost (清 history)   │
   │  ─ 首帧/重锁 → locked/relocked │
   └──────────┬─────────────────────┘
              │ 主目标 (x, y, z, range)
              ▼
   fd.update_height_history()
   ─ 逐点异常检测 (IQR + 中位数替换)
   ─ 追加到 height_history
              │
              ▼
   fd.detect_fall(height_history, settings, current_time)
   ─ 取时间窗口内的近期点
   ─ 计算: 高度下降、下降速度、加速度、低姿态持续时间
   ─ 8 项条件投票，满足 ≥ min_conditions → 报警
              │
              ▼
   ┌──────────────────┐
   │  detected: True  │──→ _show_fall_alert() + 状态面板 "Fall Alert"
   │  detected: False │──→ 状态面板 "Tracking" 或 "No Target"
   └──────────────────┘
```

### 右侧状态面板状态机

```
                  monitor_button
                   未按下
    ┌─────────────────────────────────┐
    │           Standby               │ ← 默认状态
    └─────────┬───────────────────────┘
              │ monitor_button 按下
              ▼
    ┌─────────────────────────────────┐
    │  tracked_target_state.locked?   │
    │  ├─ True  → "Tracking"         │
    │  └─ False → "No Target"        │
    └─────────┬───────────────────────┘
              │ detect_fall → detected
              ▼
    ┌─────────────────────────────────┐
    │        "Fall Alert"             │
    │  ├─ Event Snapshot 模式?        │
    │  │  └─ 自动保存特征快照         │
    │  │     → "Sample Saved"         │
    │  └─ 否 → 等待警报冷却 (3s)     │
    └─────────────────────────────────┘
```

---

## 三、核心模块详解

### `main.py` — 主界面与业务编排（~2000 行）

系统入口，承担全部编排职责：

- **配置链**: 解析 `Radar.cfg` → 生成 `runtime_cfg` → 同步给 UI / 采集 / DSP
- **采集链**: 启动 `UdpListener` + `DataProcessor` 线程
- **刷新循环**: `update_figure()` 以 ~1ms 周期执行热图刷新、点云显示、目标跟踪、跌倒检测、ML 推理、面板更新
- **GUI 布局**: 5 个 Tab（实时热图、NLOS 定位、点云、轨迹、高度检测）

关键全局变量：

| 变量 | 用途 |
|------|------|
| `tracked_target_state` | 单目标跟踪器当前状态 (`TrackerState`) |
| `height_history` | 高度历史序列 `[(timestamp, z, y, range), ...]` |
| `last_fall_detection_result` | 最近一次跌倒检测结果 |
| `_fall_predictor` | ML 预测器实例 (默认 `NullFallPredictor`) |
| `_last_ml_prediction` | 最近一次 ML 推理结果 |

---

### `DSP.py` — 雷达数字信号处理（~505 行）

两个核心函数：

| 函数 | 产出 |
|------|------|
| `RDA_Time(data, ...)` | RTI (时间-距离), DTI (多普勒-时间), RDI (距离-多普勒) |
| `Range_Angle(data, ...)` | RAI (距离-方位角), REI (距离-俯仰角) |
| `extract_pointcloud_from_angle_maps(RAI, REI, ...)` | 3D 点云 `[N, 4]` → `[range, x, y, z]` |

内部维护了 RTI/RDI/RAI/REI 的帧级滑窗队列，用于生成时间维度特征。

---

### `target_tracking.py` — 单目标跟踪器（104 行）

纯函数式设计，无副作用的状态机：

```python
TrackerState(locked, position, last_update_ms, miss_count)

update_tracker(state, candidates, current_time, gate_m, max_miss, timeout_ms)
    → (new_state, event)
    # event ∈ {"idle", "locked", "tracked", "relocked", "lost"}
```

跟踪策略：
- **首次锁定**: 取候选列表第一个（最高能量）
- **连续跟踪**: 在所有候选中找欧氏距离最近的，距离 ≤ `gate_m` 则关联
- **丢帧容忍**: `miss_count` 超过 `max_miss` 或时间超过 `timeout_ms` → 失锁
- **失锁重置**: 发生 `lost` 或 `relocked` 时，`main.py` 会清空 `height_history`

---

### `fall_detection.py` — 跌倒检测算法（~273 行）

两个公开函数：

| 函数 | 用途 |
|------|------|
| `update_height_history(history, sample, max_history)` | 追加新样本，自动做逐点异常检测和中位数替换 |
| `detect_fall(history, settings, current_time_ms)` | 基于 8 项特征条件的投票式跌倒判定 |

**检测条件（8 项投票）**：

| 条件 | 含义 |
|------|------|
| `height_drop` | 高度下降量 ≥ 阈值 |
| `velocity` | 下降速度 > 阈值 |
| `acceleration` | 加速度 > 阈值 |
| `final_height` | 终止高度 < 低姿态阈值 |
| `height_window` | 最大高度在合理窗口内 |
| `max_height_positive` | 最大高度 > 0（人在站立） |
| `min_height_low` | 最小高度足够低 |
| `low_duration` | 低姿态持续时间 > 阈值 |

灵敏度等级（`灵敏`/`中等`/`不灵敏`）控制每项阈值的具体数值和最低满足条件数。

---

### `fall_predictor.py` — ML/DL 模型接口（157 行）

```python
FallFeatureClip      # 一帧同步特征快照：RT/DT/RDT/ART/ERT + height_history + tracked_target_state
FallPrediction       # 预测结果：available, label, score, probability, topk, metadata

BaseFallPredictor    # 基类接口
NullFallPredictor    # 默认空实现（不做推理）
TorchModuleFallPredictor  # 可选 Torch 包装器

build_fall_predictor()    # 工厂函数，默认返回 NullFallPredictor
```

**设计要点**：
- `torch` 是 lazy import，不安装 Torch 也能正常运行
- ML 推理在 `update_figure()` 中以 200ms 间隔节流执行
- 预测结果只写入面板附加文本和日志，**不接管规则法报警**

---

### `pointcloud_clustering.py` — 点云聚类（342 行）

提供两种 DBSCAN 聚类模式：

| 函数 | 特点 |
|------|------|
| `cluster_pointcloud_simple(points, eps, min_samples)` | 仅用空间坐标 (x,y,z) |
| `cluster_pointcloud(points, eps, min_samples)` | 空间 + 信号强度联合聚类 |

当前 `main.py` 使用 `DBSCAN(eps=0.3, min_samples=2)` 做空间聚类。

---

### `real_time_process.py` — 数据采集与处理线程（~162 行）

| 类 | 职责 |
|------|------|
| `UdpListener` | 后台线程，通过 UDP 接收原始 ADC 数据，放入 `BinData` 队列 |
| `DataProcessor` | 后台线程，从 `BinData` 取数据，调用 `DSP.RDA_Time()` / `DSP.Range_Angle()` / `DSP.extract_pointcloud_from_angle_maps()`，结果放入 6 个 Queue |

---

### `radar_config.py` — 雷达控制

- `SerialConfig`: 通过串口发送 cfg 到雷达，控制雷达启停
- `DCA1000Config`: 配置 DCA1000 EVM 的 UDP 地址/端口

---

### `UI_interface.py` — GUI 界面定义（~1161 行）

PyQt5 界面定义，包含 5 个 Tab：

| Tab | 内容 |
|------|------|
| real-time system | 5 张热图 + 右侧控制面板（跌倒工作流 + 雷达配置 + 状态面板） |
| NLOS Localization | 预留 |
| 点云显示 | 3D 点云可视化 + 配置面板 |
| 目标运动轨迹 | 2D XY 轨迹图 |
| 高度检测 | 2D ZY 高度图 + 跌倒检测参数 + 跌倒告警标签 |

---

### `offline_feature_extractor.py` — 离线特征提取标准化 [NEW]

专门用于离线的训练数据准备，通过复用 `DSP.py` 确保特征语义与实时系统完全一致：

- **配置对齐**: 自动调用 `DSP.apply_runtime_config()`，确保 FFT 尺寸、分辨率等参数一致。
- **帧解码对齐**: 复制并封装了实时系统的原始字节解析逻辑。
- **状态对齐**: 实现了 12 帧的预热机制（Warmup），确保离线提取的累积特征（如堆叠热图）与实时运行状态一致。
- **数据输出**: 支持单帧生成器或批量保存为 `.npy` 目录结构。

### `replay_controller.py` — 回放接口预留（77 行）

仅定义了 `ReplayLoadResult` / `BaseReplaySource` / `NullReplaySource` / `ReservedBinReplaySource` 和校验逻辑，**尚未实现真正的 `.bin` 解析与回放播放**。

---

### 辅助模块

| 模块 | 职责 |
|------|------|
| `globalvar.py` | 全局变量字典（`set_value` / `get_value`），用于跨模块状态共享 |
| `colortrans.py` | Matplotlib colormap → pyqtgraph colormap 转换 |
| `iwr6843_tlv/detected_points.py` | cfg 文件解析与运行参数提取 |

---

## 四、界面与交互

### 启动方式

```powershell
python main.py
```

### 操作流程

```
1. 确认串口 → 选择 cfg → 点击 "send"
   ↓
2. 雷达启动，热图开始刷新
   ↓
3. 切换到 "点云显示" / "目标运动轨迹" / "高度检测" 页签观察
   ↓
4. 在右侧面板点击 "monitor" 按钮开始实时监测
   ↓
5. "高度检测" 页签中确认 "启用跌倒检测" 已勾选
   ↓
6. 系统进入实时跌倒监测状态，面板显示 Tracking / No Target
   ↓
7. 若检测到跌倒 → 面板显示 "Fall Alert" + 蜂鸣 + 大字提示
```

### 跌倒检测灵敏度调节

在 "高度检测" 页签右侧可选择灵敏度等级：

| 等级 | 效果 |
|------|------|
| `灵敏` | 较低阈值，更容易触发报警，适合灵敏场景 |
| `中等` | 平衡设置，推荐日常使用 |
| `不灵敏` | 较高阈值，减少误报，适合嘈杂环境 |

---

## 五、测试体系

```
tests/
├── test_cfg_runtime.py        (4 tests) — cfg 解析与运行参数提取
├── test_fall_detection.py     (5 tests) — 跌倒检测规则法（稳定站立/明显跌倒/缓慢下蹲）
├── test_fall_predictor.py     (2 tests) — ML 接口 NullFallPredictor 行为
├── test_target_tracking.py    (4 tests) — 跟踪器锁定/跟踪/丢失/重锁
└── test_replay_controller.py  — 回放接口占位
```

运行方式：

```powershell
# pytest（推荐）
python -m pytest tests/ -v

# 或 unittest
python -m unittest discover -s tests -v
```

当前测试结果：**15 passed**

---

## 六、依赖与环境

**Python**: 3.10+ 推荐（当前测试环境 3.12.7）

**核心依赖** (`requirements.txt`):

| 包 | 版本 | 用途 |
|------|------|------|
| `PyQt5` | ≥ 5.15 | GUI 框架 |
| `pyqtgraph` | ≥ 0.12 | 热图 / 2D / 3D 可视化 |
| `numpy` | ≥ 1.19 | 数值计算 |
| `torch` | ≥ 1.7 | ML 模型推理（可选，不安装不影响主功能） |
| `matplotlib` | ≥ 3.3 | Colormap 支持 |
| `pyserial` | ≥ 3.5 | 串口通信 |
| `scikit-learn` | ≥ 0.24 | DBSCAN 聚类 |
| `joblib` | ≥ 1.0 | 并行计算支持 |

安装：

```powershell
pip install -r requirements.txt
```

---

## 七、目录结构

```
iwr6843-points-visual-main/
│
├── main.py                    # 主程序入口、业务编排
├── DSP.py                     # 雷达 DSP 信号处理
├── real_time_process.py       # UDP 数据采集与处理线程
├── radar_config.py            # 雷达串口控制
│
├── fall_detection.py          # 规则法跌倒检测算法
├── target_tracking.py         # 单目标帧间跟踪器
├── fall_predictor.py          # ML/DL 模型预测接口
│
├── pointcloud_clustering.py   # 点云 DBSCAN 聚类
├── UI_interface.py            # PyQt5 界面定义
├── colortrans.py              # Colormap 转换工具
├── globalvar.py               # 全局变量管理
├── replay_controller.py       # 离线回放接口预留
├── offline_feature_extractor.py # 离线特征提取器
│
├── requirements.txt           # Python 依赖
├── README.md                  # 旧版文档（含离线训练指南）
├── PROJECT_DOC.md             # 本文档
│
├── config/
│   └── Radar.cfg              # 雷达运行时配置（唯一配置真源）
│
├── iwr6843_tlv/
│   └── detected_points.py     # cfg 解析与参数提取
│
├── tests/
│   ├── test_cfg_runtime.py
│   ├── test_fall_detection.py
│   ├── test_fall_predictor.py
│   ├── test_target_tracking.py
│   └── test_replay_controller.py
│
├── dsp/                       # DSP 辅助资源
├── dataset/                   # 数据集目录
├── firmware/                  # 固件文件
├── img/                       # 图像资源
└── libs/                      # 第三方库
```

---

## 八、模型接入指南

### 当前 ML 接口的运行方式

```
update_figure() 每帧执行
    │
    ├─ 规则法 detect_fall() → 跌倒判定 ✓ 正式报警
    │
    └─ 200ms 节流 → _fall_predictor.predict(clip) → 辅助结果
                     │
                     ├─ NullFallPredictor（默认）→ available=False，无输出
                     └─ TorchModuleFallPredictor（接入模型后）→ 面板附加显示
```

### 接入自定义模型最简示例

```python
# my_predictor.py
from fall_predictor import BaseFallPredictor, FallPrediction, FallFeatureClip

class MyFallPredictor(BaseFallPredictor):
    def __init__(self):
        import torch
        self.model = torch.jit.load('my_model.pt')
        self.model.eval()

    def predict(self, clip: FallFeatureClip) -> FallPrediction:
        import torch, numpy as np
        # 取你需要的特征
        heights = np.array([h[1] for h in clip.height_history], dtype=np.float32)
        x = torch.from_numpy(heights).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        prob = float(torch.sigmoid(out).item())
        return FallPrediction(
            available=True,
            label='fall' if prob > 0.5 else 'non-fall',
            score=prob,
            probability=prob,
        )

    def reset(self):
        pass

def build_fall_predictor():
    return MyFallPredictor()
```

### 三、推荐的离线训练链条

推荐你按下面这条线组织离线流程：

1.  **读取 `.bin`**：通过 `offline_feature_extractor.py` 进行解析。
2.  **生成特征**：复用当前项目的 `DSP.py` 逻辑（由提取器自动处理）。
3.  **输出特征**：保存为 `.npy` 片段。
4.  **组装与训练**：进行模型训练与导出。
5.  **挂回实时窗口**：在主界面加载导出的模型。

### 离线训练建议的特征对齐

离线训练时应使用与实时系统相同的 `Radar.cfg`，确保以下特征的 shape 和物理含义一致：

| 特征 | 来源 | 备注 |
|------|------|------|
| RT | `RTIData` 时间-距离 | shape 依赖 cfg 中 ADC samples 和帧数 |
| DT | `DTIData` 多普勒-时间 | |
| RDT | `RDIData` 距离-多普勒 | |
| ART | `RAIData` 距离-方位角 | |
| ERT | `REIData` 距离-俯仰角 | |
| `height_history` | 主目标高度序列 | `[(timestamp_ms, z, y, range), ...]` |

---

## 九、当前版本的边界与限制

| 限制 | 说明 |
|------|------|
| 单人场景 | 跟踪器按单目标设计，多人场景可能目标跳变 |
| 规则法主导 | ML 推理不参与正式报警决策 |
| 回放未实现 | `.bin` 回放仅完成接口预留 |
| GUI 线程推理 | ML 推理在主线程执行，复杂模型可能造成卡顿 |
| 高度精度受限 | 受 DBSCAN 聚类精度和雷达分辨率影响 |

---

## 十、后续推荐方向

1. **.bin 回放链完成** — 让离线数据能通过 `replay_controller.py` 驱动整条 DSP 链
2. **离线特征提取标准化** — 复用 `DSP.py` 生成与实时一致的训练特征
3. **模型训练与部署** — 导出 TorchScript 或 `.py` 插件，通过 `build_fall_predictor()` 接入
4. **ML 推理线程化** — 将 Torch 推理移到独立线程，避免阻塞 GUI
5. **多人扩展** — 将 `target_tracking.py` 扩展为多目标跟踪器
6. **模型参与报警策略** — 在规则法基础上叠加模型置信度作为联合决策
