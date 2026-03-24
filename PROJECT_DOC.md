# IWR6843 实时跌倒检测与可视化系统 — 项目全景文档

> **版本日期**: 2026-03-24  
> **定位**: 基于 TI IWR6843 毫米波雷达的单目标实时跌倒检测系统  
> **当前主架构**: `Radar.cfg → UDP 采集 → 训练对齐 RD/RA/RE → 当前帧点云 → 单目标 Kalman 跟踪 → MLAlarmState → Fall Alert`

本文档是对当前仓库的全景说明。  
如果你只想快速上手，请先看 [README.md](README.md)。  
如果你只关心实时主链路，请直接看 [REALTIME_FALL_DETECTION_CHAIN.md](REALTIME_FALL_DETECTION_CHAIN.md)。

---

## 目录

- [一、项目当前到底是什么](#一项目当前到底是什么)
- [二、20 分钟阅读路径](#二20-分钟阅读路径)
- [三、整体架构与数据流](#三整体架构与数据流)
- [四、实时链路详解](#四实时链路详解)
- [五、离线训练与部署链路](#五离线训练与部署链路)
- [六、核心模块说明](#六核心模块说明)
- [七、测试与环境](#七测试与环境)
- [八、当前限制](#八当前限制)
- [九、建议的下一步](#九建议的下一步)

---

## 一、项目当前到底是什么

这个仓库已经不再是早期那种“雷达可视化 + 规则实验 + 采样脚本混在一起”的实验工程，而是更清晰地收敛成了两条主线：

1. **实时主线**
   - 用 `config/Radar.cfg` 驱动运行时参数
   - 从实时雷达流中提取训练对齐的 `RD / RA / RE`
   - 从当前帧点云提主目标，再用单目标 Kalman Filter 跟踪
   - 由 `runtime/ml_alarm.py` 维护的 ML 告警状态机决定最终 `Fall Alert`

2. **离线主线**
   - 从 `.bin` 生成 JSONL 标注
   - 提取离线 `RD / RA / RE / PC`
   - 构建训练 manifest
   - 训练 RACA 模型
   - 打包模型与 `model_meta.json`
   - 回挂到实时 GUI

### 当前能力概览

| 能力 | 当前状态 | 备注 |
| --- | --- | --- |
| `Radar.cfg` 运行时配置化 | 已实现 | `config/Radar.cfg` 是当前配置真源 |
| 实时 UDP 采集与帧重组 | 已实现 | 依赖 `libs/UDPCAPTUREADCRAWDATA.dll` |
| 训练对齐 `RD / RA / RE` 实时特征链 | 已实现 | 实时与离线共用 `runtime/aligned_features.py` |
| 当前帧点云提取 + DBSCAN 聚类 | 已实现 | 聚类参数来自 UI |
| 单目标 Kalman 跟踪 | 已实现 | 支持稳定锁存、短暂预测、近距重锁 |
| ML 主导的实时跌倒告警 | 已实现 | 由 `runtime/ml_alarm.py` 决定 |
| 模型加载 `.pth / .pt / .jit / .ts / .py` | 已实现 | 支持 RACA checkpoint 与插件 |
| `model_meta.json` 契约校验 | 已实现 | 与 `Radar.cfg` 绑定 |
| 离线 `.bin -> 特征 -> 训练 -> 打包` | 已实现 | 已形成闭环 |
| GUI 中真实 `.bin` 回放 | 未实现 | 当前仍为占位接口 |

### 需要特别澄清的三点

1. **实时正式告警已经不是规则法主导**
   - 当前真正触发 `_show_fall_alert()` 的主链是：  
     `模型推理 -> MLAlarmState -> Fall Alert`

2. **`.pth` 已经可以直接加载**
   - 这和较早版本说明不同
   - 当前 `.pth` 会走 `runtime.raca_predictor.RACAfallPredictor`

3. **雷达启动后会自动进入监测态**
   - 不再要求用户每次再手动把首页长期从 `Standby` 切出来

---

## 二、20 分钟阅读路径

如果你想快速理解这个项目，推荐这样读：

| 阶段 | 时间 | 阅读位置 | 目标 |
| --- | --- | --- | --- |
| 1 | 3 min | 本文档的 [三、整体架构与数据流](#三整体架构与数据流) | 建立全局图 |
| 2 | 4 min | [README.md](README.md) | 先抓住当前能力边界 |
| 3 | 5 min | `main.py` 开头、`_build_runtime_workers()`、`_consume_ml_feature_frames()` | 理解实时编排 |
| 4 | 3 min | `real_time_process.py` + `runtime/aligned_features.py` | 理解实时特征从哪来 |
| 5 | 3 min | `runtime/target_tracking.py` | 理解目标如何稳定 |
| 6 | 2 min | `runtime/ml_alarm.py` | 理解正式告警如何触发 |

如果你是为了训练模型并上线：

1. 先看 [README.md](README.md) 里的离线流程
2. 再看 `offline/feature_extractor.py`
3. 再看 `training/train_v2.py`
4. 最后看 `training/export_model.py`

---

## 三、整体架构与数据流

### 全局数据流

```text
Radar.cfg
   ↓
main.py 解析运行时参数
   ↓
UdpListener + DataProcessor
   ↓
runtime.aligned_features.extract_training_aligned_frame_features()
   ↓
RD / RA / RE
   ├─→ MLFeatureData 队列 → 模型推理 → MLAlarmState → Fall Alert
   └─→ 当前帧点云提取 → DBSCAN 聚类 → 单目标 Kalman 跟踪
```

### 启动阶段做了什么

当前 `main.py` 的启动关键步骤可以概括为：

1. 读取当前 UI 选择的 `cfg / CLI / Data port`
2. 解析并应用 `Radar.cfg`
3. 构建实时 worker：
   - `UdpListener`
   - `DataProcessor`
4. 发送配置并启动雷达
5. 启动刷新循环
6. 调用 `_set_monitor_enabled(True)` 自动进入监测态

### 当前主配置真源

当前仓库明确把：

```text
config/Radar.cfg
```

作为实时和离线都尽量要对齐的配置真源。

这份 cfg 会影响：

- `num_range_bins`
- `num_doppler_bins`
- `frame_length`
- `frame_periodicity_ms`
- 角度分辨率与角度 bins

所以它不仅影响实时 DSP，也直接影响训练输入 shape 和 `model_meta.json` 契约。

---

## 四、实时链路详解

### 1. 实时特征链

当前实时处理已经和训练特征对齐。

`real_time_process.DataProcessor` 对每一帧调用：

```python
runtime.aligned_features.extract_training_aligned_frame_features(...)
```

直接得到：

- `RD`
- `RA`
- `RE`

然后：

- `RD / RA / RE` 进入 `MLFeatureData`
- `RA / RE` 同时用于当前帧点云提取

这意味着当前主链路不再依赖旧版显示专用的那套“先算热图、再从显示图回推逻辑”的路径。

### 2. 点云与聚类

当前主目标候选来自**当前帧点云**，不是历史融合点云。

流程是：

1. 从当前帧 `RA / RE` 直接提点云
2. 按当前 UI 的 `eps / min_samples` 做 `DBSCAN`
3. 过滤点数不足或距离过近的簇
4. 按簇能量排序
5. 生成候选目标中心送给 tracker

### 3. 单目标 Kalman 跟踪

当前 tracker 位于：

```text
runtime/target_tracking.py
```

核心特征：

- 单目标
- 常速度 Kalman Filter
- 状态向量：`[x, y, z, vx, vy, vz]`
- 观测向量：`[x, y, z]`
- 支持事件：
  - `idle`
  - `locked`
  - `tracked`
  - `predicted`
  - `relocked`
  - `lost`

当前实时主链依赖的几个默认门限：

- `stable_hits = 2`
- `max_miss = 2`
- `timeout_ms = 2000`
- `relock_grace_ms = 1000`
- `relock_distance_m = 0.6`

### 4. 实时告警现在由谁决定

当前正式告警由：

```text
runtime/ml_alarm.py
```

中的 `MLAlarmState` 决定。

它会综合考虑：

- 监测是否开启
- 模型是否已加载
- 模型契约是否通过校验
- 当前推理是否为 `fall`
- 概率是否超过阈值
- 连续阳性次数是否达到要求
- tracker 是否已进入稳定门控
- 当前是否 lagging

### 5. `Standby`、`No Target`、`Tracking` 现在是什么意思

首页主状态现在更接近“监测状态 + 目标状态 + ML 状态”的组合，而不是单纯规则法状态。

主要状态含义：

- `Standby`
  - 监测未开启
  - 或雷达未启动/启动失败
- `No Target`
  - 已开启监测，但没有可用目标
- `Target Warmup`
  - 有锁定目标，但还没达到稳定门控
- `Tracking`
  - 已有稳定目标
- `Fall Alert`
  - ML 告警状态机正式确认跌倒

如果模型没有加载，或契约校验失败，页面不会自动退回 `Standby`，而是会在状态卡/技术摘要里显示：

```text
ML disabled
```

### 6. 规则法模块当前还在做什么

`runtime/fall_detection.py` 仍然保留在仓库里，但当前定位更偏向：

- 历史算法保留
- 高度序列辅助
- 调试和对照分析

它已经不再是实时正式告警的主入口。

换句话说：

- 规则法还“在仓库里”
- 但当前正式跌倒提示不是靠它触发

---

## 五、离线训练与部署链路

当前仓库已经具备一条比较完整的离线闭环：

```text
.bin
  ↓
training.generate_label_template / training.split_adl5min
  ↓
offline.feature_extractor
  ↓
training.build_manifest
  ↓
training.train_v2
  ↓
training.export_model
  ↓
实时 GUI 加载
```

### 1. 生成标注模板

适用脚本：

- `training/generate_label_template.py`
- `training/split_adl5min.py`

前者适合标准命名的原始 `.bin` 文件，后者适合 `ADL_5min` 这种跨多个 `Raw_N.bin` 的长时非跌倒数据。

### 2. 提取离线特征

适用脚本：

```text
offline/feature_extractor.py
```

输出标准目录：

```text
features/<label>/<clip_id>/
├─ RD.npy
├─ RA.npy
├─ RE.npy
├─ PC.npy
└─ meta.json
```

这里最重要的是：离线提取和实时推理共用同一套 `runtime/aligned_features.py` 契约。

### 3. 构建训练 manifest

适用脚本：

```text
training/build_manifest.py
```

它会扫描已提取的样本目录，生成训练用 `JSONL`。

### 4. 训练模型

当前更推荐的训练入口：

```text
training/train_v2.py
```

当前项目更偏向下面这套配置：

- `fall` vs `non-fall` 二分类
- 小数据量时优先 `LSTM`
- `RE` 分支对 IWR6843ISK 建议降权

默认训练产物通常会包含：

- `raca_v2_best.pth`
- `raca_v2_last.pth`
- `history.json`
- `training_curves.png`
- `confusion_matrix.png`
- `test_report.txt`

### 5. 打包模型给实时程序

适用脚本：

```text
training/export_model.py
```

这个步骤会：

- 复制模型文件
- 生成 `model_meta.json`
- 记录 `cfg_sha256`
- 记录 `clip_frames`
- 记录 `frame_periodicity_ms`
- 记录 `feature_shapes`

### 6. 为什么 `model_meta.json` 很重要

运行时加载模型后，`runtime/fall_predictor.py` 会做契约校验。

它会检查：

- 当前运行时 `cfg`
- 模型声明的 `feature_shapes`
- `clip_frames`
- `frame_periodicity_ms`
- `cfg_sha256`

如果不一致，当前实时程序会显示：

```text
ML disabled: cfg mismatch
```

这也是为什么训练、打包、部署必须尽量使用同一份 `Radar.cfg`。

---

## 六、核心模块说明

### `main.py`

项目主入口与总编排层，负责：

- 启动 Qt 界面
- 解析并应用 `Radar.cfg`
- 构建实时 worker
- 刷新热图和状态板
- 消费 `MLFeatureData`
- 驱动 `MLAlarmState`
- 管理模型加载、回放入口与状态显示

### `real_time_process.py`

实时采集与处理线程层，包含：

- `UdpListener`
- `DataProcessor`

当前 `DataProcessor` 的关键职责是：

- 从 DLL 缓冲区读取原始 IQ
- 还原一帧复数雷达数据
- 提取训练对齐的 `RD / RA / RE`
- 生成当前帧点云
- 把结果送入队列

### `runtime/aligned_features.py`

这是实时与离线共用的特征契约中心。

它定义了：

- 角度 bins 如何根据 cfg 计算
- 训练时的特征 shape
- 单帧 `RD / RA / RE` 的提取方式

### `runtime/target_tracking.py`

单目标 Kalman Filter 跟踪器。

它负责：

- 首次锁定
- 连续跟踪
- 预测态保持
- 失锁判定
- 近距重锁继承

### `runtime/ml_alarm.py`

正式告警状态机。

它负责：

- ML 是否启用
- 连续阳性计数
- threshold / required_streak
- tracker_ready 门控
- lagging 与 error 状态

### `runtime/fall_predictor.py`

模型加载与契约校验总入口。

当前支持：

- `.py` 插件
- `.pt / .jit / .ts`
- `.pth`

同时也负责：

- 自动发现默认模型
- 读取 `model_meta.json`
- 校验 `cfg_sha256`

### `runtime/raca_predictor.py`

RACA checkpoint 的实时适配器。

当前默认使用：

- `window = 100`
- `stride = 10`

### `offline/feature_extractor.py`

离线特征提取器。

职责是：

- 解析 `.bin`
- 复用 cfg
- 生成 `RD / RA / RE / PC`
- 写出 `meta.json`

### `training/*.py`

训练相关辅助脚本主要包括：

- `generate_label_template.py`
- `split_adl5min.py`
- `build_manifest.py`
- `train_v2.py`
- `export_model.py`

### `runtime/replay_controller.py`

当前还是回放占位接口。

已实现：

- `.bin` 文件选择与登记
- 基本文件合法性检查

未实现：

- `.bin` 真正流式解析
- 回放驱动实时 GUI

---

## 七、测试与环境

### 测试入口

当前建议的基础测试命令：

```powershell
python -m unittest discover -s tests -v
```

覆盖方向主要包括：

- cfg 运行时解析
- 实时特征与处理线程
- tracker
- ML alarm
- predictor 加载与契约
- offline extractor
- export model
- replay placeholder

### 运行环境

推荐：

- Python `3.10`
- Windows
- 已安装 `requirements.txt` 中依赖

实时链还依赖：

- `libs/UDPCAPTUREADCRAWDATA.dll`
- TI IWR6843
- DCA1000

---

## 八、当前限制

| 限制 | 说明 |
| --- | --- |
| 单目标优先 | 当前 tracker 不是多目标方案 |
| 回放未接通 | `.bin` 回放还只是占位 |
| 模型契约严格 | cfg 或 feature shape 不一致会被禁用 |
| TorchScript 适配器偏通用 | 复杂输入更适合 `.py` 插件或 `.pth + RACA` |
| GUI 运行仍偏重实时现场使用 | 不是一个独立的训练平台 |

---

## 九、建议的下一步

如果沿着当前架构继续推进，最自然的方向是：

1. **补全真实 `.bin` 回放链**
   - 让离线数据能驱动实时 GUI

2. **继续打磨离线数据规范**
   - 统一标注命名
   - 统一 clip 切分方式

3. **完善模型产物规范**
   - 让训练、打包、部署的 `model_meta.json` 约束更稳定

4. **评估多人扩展**
   - 当前如果要上多人场景，需要从 tracker 开始重新设计

5. **继续分离“实时产品逻辑”和“研究逻辑”**
   - 当前仓库已经做了大量收敛，后面继续保持这种边界会更好维护

---

## 总结

当前这个仓库已经形成了一条很明确的闭环：

- 实时侧：`RD / RA / RE + 当前帧点云 + 单目标跟踪 + ML 告警`
- 离线侧：`.bin -> 特征 -> manifest -> 训练 -> 打包`
- 中间靠：`Radar.cfg + model_meta.json` 做契约绑定

如果你的目标是“用离线 `.bin` 训练一个模型，再把它稳定挂回实时 GUI”，当前结构已经足够作为后续工作的稳固基础。
