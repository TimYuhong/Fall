# IWR6843 实时跌倒检测与可视化系统

本文档按 **2026-03-24** 的仓库状态整理。

这是一个围绕 TI IWR6843 毫米波雷达构建的实时跌倒检测项目。当前仓库已经从早期的“可视化/采样实验工程”收敛为一套更明确的链路：

- 统一使用 `config/Radar.cfg` 作为运行时配置真源
- 实时提取与训练对齐的 `RD / RA / RE` 特征
- 从当前帧点云做聚类，并用单目标 Kalman Filter 跟踪主目标
- 由 **ML 告警状态机** 驱动最终 `Fall Alert`
- 支持离线 `.bin` 特征提取、训练、模型打包与实时加载

如果你需要更细的实时链路说明，可以继续看 [REALTIME_FALL_DETECTION_CHAIN.md](REALTIME_FALL_DETECTION_CHAIN.md)。

## 当前状态

| 能力 | 当前状态 |
| --- | --- |
| 实时采集与可视化 | 已实现 |
| 训练对齐 `RD / RA / RE` 实时特征链 | 已实现 |
| 当前帧点云提取 + DBSCAN 聚类 | 已实现 |
| 单目标 Kalman 跟踪 | 已实现 |
| ML 主导的实时跌倒告警 | 已实现 |
| 加载 `.pth / .pt / .jit / .ts / .py` 模型 | 已实现 |
| `model_meta.json` 契约校验与 `Radar.cfg` 绑定 | 已实现 |
| 离线 `.bin -> 特征 -> manifest -> 训练 -> 打包` 工具链 | 已实现 |
| GUI 中真正的 `.bin` 回放播放 | 尚未接通，仅预留入口 |

## 这版 README 重点说明什么

相较于仓库中较早的说明，这里有几处需要特别澄清：

- 当前实时主告警已经是 **ML 主导**，`runtime/ml_alarm.py` 决定最终 `Fall Alert`
- `runtime/fall_detection.py` 仍保留在仓库中，但**不再是实时主链路的最终告警源**
- 当前代码已经支持直接加载 **RACA checkpoint `.pth`**
- 雷达成功启动后，界面会**自动进入监测态**，不再长期停在 `Standby`

## 仓库结构

```text
.
├─ config/
│  └─ Radar.cfg
├─ runtime/
│  ├─ aligned_features.py     # 实时/离线共用的 RD/RA/RE 提取契约
│  ├─ target_tracking.py      # 单目标 Kalman 跟踪
│  ├─ ml_alarm.py             # ML 告警状态机
│  ├─ fall_predictor.py       # 模型加载、契约校验、通用预测接口
│  ├─ raca_predictor.py       # RACA checkpoint 实时推理适配
│  ├─ replay_controller.py    # .bin 回放占位接口
│  └─ radar_config.py         # CLI / DCA1000 控制
├─ offline/
│  └─ feature_extractor.py    # 离线 .bin 特征提取
├─ training/
│  ├─ generate_label_template.py
│  ├─ split_adl5min.py
│  ├─ build_manifest.py
│  ├─ train_v2.py
│  └─ export_model.py
├─ tests/
├─ main.py                    # 主界面与实时编排
├─ real_time_process.py       # UDP 采集与实时帧处理
├─ DSP.py
├─ UI_interface.py
├─ PROJECT_DOC.md
└─ REALTIME_FALL_DETECTION_CHAIN.md
```

## 环境与依赖

### 运行环境建议

- 建议 Python `3.10`
- 实时采集链依赖 `libs/UDPCAPTUREADCRAWDATA.dll`，因此更适合在 **Windows** 环境运行
- 实时运行通常需要：
  - TI IWR6843 雷达
  - DCA1000
  - 正确的 CLI/Data 串口与网络配置

### 安装依赖

```powershell
pip install -r requirements.txt
```

当前 `requirements.txt` 主要包含：

- `PyQt5`
- `pyqtgraph`
- `numpy`
- `torch`
- `matplotlib`
- `pyserial`
- `scikit-learn`
- `joblib`

## 快速开始

### 1. 启动实时界面

```powershell
python main.py
```

### 2. 选择并发送配置

默认运行配置为：

```text
config/Radar.cfg
```

在界面中确认：

- `cfg` 路径
- CLI 串口
- Data 串口

然后点击发送/启动相关按钮。

### 3. 启动后的当前行为

一旦雷达成功启动，当前代码会：

1. 解析并应用 `Radar.cfg`
2. 启动 `UdpListener` 与 `DataProcessor`
3. 开始实时提取训练对齐特征
4. 自动切换到监测态

也就是说，`Standby` 主要只会出现在这些场景：

- 雷达尚未启动
- 启动失败
- 用户手动关闭监测

如果模型未加载，或者模型契约校验失败，界面仍然可以进入监测态，但 ML 告警会显示为：

```text
ML disabled
```

### 4. 模型加载

当前界面支持加载：

- `*.pth`：RACA checkpoint
- `*.pt` / `*.jit` / `*.ts`：TorchScript
- `*.py`：自定义预测器插件

如果界面中的模型路径为空，程序还会尝试自动发现默认模型：

- 环境变量 `RADAR_MODEL_PATH`
- 环境变量 `RADAR_DEFAULT_MODEL_PATH`
- `checkpoints/**/raca_v2_best.pth`
- `checkpoints/**/raca_best.pth`

### 5. 回放入口

界面中可以选择 `.bin` 文件作为回放源，但当前 `runtime/replay_controller.py` 仍然只是占位实现：

- 可以登记 `.bin` 文件
- 还**不能**把 `.bin` 真正流式回放到实时 GUI

## 当前实时链路

当前实时主链路已经对齐到训练特征，不再以旧的显示链作为告警主输入。

### 1. 特征提取

`real_time_process.DataProcessor` 会对每帧调用：

```python
runtime.aligned_features.extract_training_aligned_frame_features(...)
```

得到与离线训练一致的三类特征：

- `RD`
- `RA`
- `RE`

这三类特征同时服务两件事：

- 送入 `MLFeatureData` 队列给模型推理
- 从当前帧 `RA / RE` 中直接提取点云

### 2. 点云、聚类与主目标跟踪

当前跟踪链使用的是**当前帧点云**，而不是历史衰减后的融合点云。

主流程如下：

1. 从当前帧 `RA / RE` 提取点云
2. 对当前帧点云做 `DBSCAN` 聚类
3. 生成候选目标中心
4. 交给 `runtime.target_tracking.update_tracker()` 进行关联与跟踪

当前 tracker 的核心特征：

- 单目标
- 常速度 Kalman Filter
- 状态量：`[x, y, z, vx, vy, vz]`
- 支持 `locked / tracked / predicted / relocked / lost`
- 支持稳定态迟滞与近距重锁继承

从当前测试与代码默认值看，实时跟踪的重要门限包括：

- `stable_hits = 2`
- `max_miss = 2`
- `timeout_ms = 2000`
- `relock_grace_ms = 1000`
- `relock_distance_m = 0.6`

### 3. ML 告警状态机

最终 `Fall Alert` 由 `runtime/ml_alarm.py` 维护的状态机驱动，而不是旧版规则法直接驱动。

它会综合考虑：

- 模型是否已加载
- 模型契约是否通过校验
- tracker 是否处于可用/稳定状态
- 当前推理是否为正类
- 概率阈值
- 连续阳性次数
- 是否出现推理滞后

仓库内置的预设策略为：

- `灵敏`：`threshold=0.50`, `required_streak=1`
- `中等`：`threshold=0.50`, `required_streak=2`
- `稳健`：`threshold=0.65`, `required_streak=2`

### 4. 规则法模块的当前位置

`runtime/fall_detection.py` 仍在仓库中保留，但当前实时主链路里，最终告警已经不再直接调用它。

如果你正在看旧文档或旧分支，需要特别注意这一点：

- 现在的正式 `Fall Alert` 以 **ML 告警状态** 为准
- 规则法更接近“历史模块/辅助逻辑保留”，而不是主报警入口

## 模型加载与契约校验

### 支持的模型形式

#### 1. RACA checkpoint：`.pth`

这是当前仓库里最直接、最匹配实时链路的方式。

- 由 `runtime.raca_predictor.RACAfallPredictor` 加载
- 默认使用滑窗：
  - `window = 100`
  - `stride = 10`
- 与实时 `RD / RA / RE` 队列直接对接

#### 2. TorchScript：`.pt / .jit / .ts`

支持直接加载，但要注意：

- 当前默认包装器只是一个简单的通用适配器
- 对复杂多输入模型，通常更推荐用 `.py` 插件或直接使用 `.pth + RACA` 方案

#### 3. Python 插件：`.py`

插件文件可以导出以下任一入口：

- `build_fall_predictor()`
- `build_predictor()`
- `PREDICTOR`

并最终返回 `runtime.fall_predictor.BaseFallPredictor` 的实例。

### `model_meta.json`

模型打包后，可以在模型目录旁放置 `model_meta.json`。当前代码会读取并校验：

- `clip_frames`
- `frame_periodicity_ms`
- `feature_shapes`
- `class_names`
- `positive_labels`
- `cfg_sha256`

如果运行时配置与模型契约不一致，状态会变成：

```text
ML disabled: cfg mismatch
```

这也是为什么**训练、打包、实时运行必须使用同一份 `Radar.cfg`**。

## 离线数据与训练流程

当前仓库已经具备一条完整的离线工具链：

```text
.bin -> 标注 manifest -> 特征提取 -> 训练 manifest -> 模型训练 -> 模型打包 -> 实时加载
```

### 1. 从原始 `.bin` 自动生成标注模板

如果你的原始数据文件名符合约定，例如：

```text
fall_S01_dormitory_02_Raw_0.bin
walk_S03_corridor_01_Raw_0.bin
```

可以直接生成 JSONL：

```powershell
python -m training.generate_label_template `
  --data-root dataset\raw `
  --out dataset\labels.jsonl
```

这个脚本会从文件名中解析：

- `label`
- `subject_id`
- `session_id`
- `scene`

### 2. ADL_5min 多段数据切片

如果你有 `ADL_5min` 这类被分成多个 `Raw_N.bin` 的长时非跌倒数据，可以使用：

```powershell
python -m training.split_adl5min `
  --adl-root F:\Data_bin\bedroom\ADL_5min `
  --out dataset\adl_non_fall.jsonl `
  --segment-frames 100 `
  --label non_fall
```

### 3. 提取离线特征

```powershell
python -m offline.feature_extractor `
  --manifest dataset\labels.jsonl `
  --out dataset\features
```

单文件调试模式也支持：

```powershell
python -m offline.feature_extractor `
  --bin F:\Data_bin\demo.bin `
  --label debug `
  --out dataset\debug_features
```

默认输出目录结构为：

```text
dataset/features/<label>/<clip_id>/
├─ RD.npy
├─ RA.npy
├─ RE.npy
├─ PC.npy
└─ meta.json
```

其中当前训练契约的主要 shape 为：

- `RD`: `(T, R, D)`
- `RA`: `(T, A, R)`
- `RE`: `(T, A, R)`

默认情况下，`T` 通常是 `100` 帧。

### 4. 构建训练 manifest

```powershell
python -m training.build_manifest `
  --features-root dataset\features `
  --out dataset\train_manifest.jsonl
```

### 5. 训练模型

当前更推荐使用：

```powershell
python -m training.train_v2 `
  --manifest dataset\train_manifest.jsonl `
  --out-dir checkpoints `
  --binary `
  --force-model lstm `
  --re-weight 0.3
```

这条命令对应的是当前仓库更推荐的二分类跌倒检测配置：

- `fall` vs `non-fall`
- 小数据集优先 `LSTM`
- IWR6843ISK 场景下 `RE` 分支建议降权 `0.3`

训练完成后，`train_v2.py` 会在输出目录下生成类似：

- `raca_v2_best.pth`
- `raca_v2_last.pth`
- `history.json`
- `training_curves.png`
- `confusion_matrix.png`
- `test_report.txt`

### 6. 打包模型供实时程序使用

```powershell
python -m training.export_model `
  --model checkpoints\your_run\raca_v2_best.pth `
  --out deploy\fall_v1 `
  --cfg config\Radar.cfg `
  --sample-meta dataset\features\fall\clip_001\meta.json `
  --class-names fall non_fall `
  --positive-labels fall
```

这个步骤会：

- 复制模型文件到输出目录
- 生成 `model_meta.json`
- 固化当前 `Radar.cfg` 的 `sha256`
- 固化 `clip_frames / frame_periodicity_ms / feature_shapes`

之后你可以：

- 直接在界面中手动选择该模型
- 或把模型放在 `checkpoints/` 下，让程序自动发现

## 测试

运行全部单元测试：

```powershell
python -m unittest discover -s tests -v
```

当前测试主要覆盖：

- `Radar.cfg` 运行时解析
- `runtime.aligned_features`
- `runtime.target_tracking`
- `runtime.ml_alarm`
- `runtime.fall_predictor`
- `offline.feature_extractor`
- `training.export_model`
- `real_time_process`
- 回放占位接口

## 当前限制

- GUI 中的 `.bin` 回放还没有真正接通，只是保留了入口
- 当前实时跟踪是**单目标优先**设计，不保证多人体场景下的稳定性
- 如果没有加载模型，或契约校验失败，ML 告警会被禁用
- TorchScript 的默认适配器偏通用；对于复杂输入契约，更推荐 `.py` 插件或 `.pth` RACA 方案
- 离线提取、训练、打包、实时运行必须尽量保持同一份 `config/Radar.cfg`

## 详细文档

- [REALTIME_FALL_DETECTION_CHAIN.md](REALTIME_FALL_DETECTION_CHAIN.md)：当前实时主链路、tracker、ML 告警细节
- [PROJECT_DOC.md](PROJECT_DOC.md)：更长篇的项目背景与整理说明

## 总结

当前仓库已经具备一条比较清晰的闭环：

1. 实时侧用 `RD / RA / RE + 点云 + 单目标跟踪 + ML 告警`
2. 离线侧用 `.bin -> 特征 -> 训练 -> 打包`
3. 通过 `model_meta.json + Radar.cfg` 把训练契约和实时契约绑在一起

如果你的目标是“基于离线 `.bin` 训练模型，再挂回实时 GUI 做在线跌倒检测”，这套结构已经可以直接继续往下推进。
