# 实时跌倒检测链路

## 结论

当前实现已经是 **ML 主导的实时跌倒检测**：

- 最终 `Fall Alert` 由深度学习模型输出驱动。
- 规则法 `detect_fall()` 已经从实时主链路解绑，不再参与正式告警。
- 目标跟踪采用 **当前帧聚类 + 常速度 Kalman Filter + 稳定迟滞/近距重锁继承**。
- 雷达启动成功后会 **自动进入监测态**，不再默认长期停在 `Standby`。
- `real-time` 首页采用 **上方固定检测看板 + 下方滚动概率曲线** 的布局。

## 总流程

```mermaid
flowchart TD
    A[sendconfigfunc] --> B[openradar]
    B --> C[apply_runtime_cfg]
    B --> D[_build_runtime_workers]
    D --> E[UdpListener]
    D --> F[DataProcessor]
    B --> G[_set_monitor_enabled(True)]

    F --> H[训练对齐单帧特征 RD/RA/RE]
    H --> I[MLFeatureData queue]
    H --> J[点云提取]

    J --> K[current_frame_pointcloud]
    K --> L[DBSCAN 当前帧聚类]
    L --> M[target_tracking.update_tracker]
    M --> N[stable_latched / predicted / relocked / lost]

    I --> O[_consume_ml_feature_frames]
    O --> P[RACAfallPredictor]
    P --> Q[FallPrediction]
    Q --> R[MLAlarmState]

    N --> R
    R --> S[_show_fall_alert]
    R --> T[_update_realtime_detection_dashboard]
    N --> T
    T --> U[real-time 首页状态]
```

## 1. 启动与配置链路

### 启动入口

1. `sendconfigfunc()` 读取当前 UI 中选择的 `cfg / CLI port / Data port`
2. 调用 `openradar(config, com, data_port)`
3. `openradar()` 内部顺序：
   - `apply_runtime_cfg()`
   - `SerialConfig.StopRadar()`
   - `_build_runtime_workers()`
   - `collector.start()`
   - `processor.start()`
   - `SendConfig()`
   - `StartRadar()`
4. `capture_started = True` 后调用 `update_figure()`
5. 立即调用 `_set_monitor_enabled(True)`，自动进入监测态

### 当前行为

- `Standby` 只在这些场景出现：
  - 用户手动关闭监测
  - 雷达未启动
  - 启动失败
- 只要雷达成功 `sensorStart` 且 worker 正常启动，首页就会自动切到监测态

## 2. 原始数据采集与特征生成

### 实时采集

- `UdpListener` 持续接收雷达 UDP 数据
- `DataProcessor.run()` 从 DLL 缓冲区中读取原始 `int16 IQ`
- 经过重排后恢复成当前帧的复数雷达数据

### 实时特征链

当前 `DataProcessor` 只保留一条必要计算链：

1. 对当前帧调用训练对齐特征提取
2. 得到单帧：
   - `RD`
   - `RA`
   - `RE`
3. 这 3 个特征同时服务两件事：
   - 送入 `MLFeatureData` 队列给模型
   - 从对齐后的 `RA / RE` 直接提点云

### 已经裁掉的旧链路

下面这些旧的显示专用重计算链已经不再是实时主链：

- `DSP.RDA_Time()`
- 旧版显示用 `RTI / RDI / DTI`
- 旧版单独重算的 `RAI / REI`

现在保留的特征图来源只有训练对齐后的 `RD / RA / RE`。

## 3. 点云与当前帧候选

### 点云来源

- 点云由当前帧的对齐 `RA / RE` 直接提取
- 不再从历史热图或显示热图回推

### 当前帧聚类

实时主目标跟踪只使用 **当前帧新点云**，不再在历史衰减合并点云上锁目标。

聚类步骤：

1. 读取 `current_frame_pointcloud`
2. 取空间坐标 `[x, y, z]`
3. 用 `DBSCAN` 聚类

当前聚类参数来源：

- `eps = pointcloud_cluster_eps.value()`
- `min_samples = max(4, pointcloud_cluster_min_samples.value())`

这里的 UI 数值是当前唯一真源，不再在跟踪链里硬编码另一套 `eps=0.3 / min_samples=2`。

### 候选过滤

每个簇会先过滤，再送进 tracker：

- 点数不足直接丢弃
- `range <= 0.25m` 的近场簇直接丢弃
- 用 `1 / (range + 0.01)` 做能量权重
- 加权得到簇中心 `(x, y, z, range)`
- 再按簇能量排序

## 4. 目标跟踪逻辑

## 跟踪器类型

当前 tracker 是 **单目标常速度 Kalman Filter**。

### 状态向量

`[x, y, z, vx, vy, vz]`

### 观测向量

`[x, y, z]`

### 跟踪流程

每次 `update_tracker()` 的流程固定为：

1. `predict(dt)`
2. 用预测位置对当前帧候选做关联
3. 若匹配成功，执行 `update(z)`
4. 若匹配失败，进入 `predicted`
5. 若连续丢失超过阈值或超时，进入 `lost`

### 关联门控

关联时同时使用：

- 卡尔曼预测协方差带来的马氏距离门控
- 欧氏距离硬门限 `gate_m`

### TrackerState 关键字段

- `locked`
- `position`
- `predicted_position`
- `velocity`
- `covariance`
- `last_update_ms`
- `last_measurement_ms`
- `miss_count`
- `hit_streak`
- `age_frames`
- `status`
- `stable_latched`
- `last_stable_position`
- `last_stable_time_ms`

### 事件语义

当前对外事件包括：

- `idle`
- `locked`
- `tracked`
- `predicted`
- `relocked`
- `lost`

## 5. 稳定目标定义

### 稳定阈值

当前稳定策略是“均衡稳态”：

- 新锁定目标在 `hit_streak >= 2` 时进入稳定态
- 一旦 `stable_latched=True`，后续短暂丢帧不会立刻失去门控

### 短暂闪断

当 tracker 处于：

- `predicted`
- 且 `miss_count <= 2`
- 且 `stable_latched=True`

此时仍然视为 **稳定目标有效**。

这就是现在 ML 连续阳性不会因为 1-2 帧点云闪断被直接清空的原因。

### 近距重锁继承

当发生真正 `lost` 后，如果：

- 在 `1000ms` 内重锁
- 与 `last_stable_position` 欧氏距离 `<= 0.6m`

则按“同一目标回归”处理：

- 事件仍是 `relocked`
- `hit_streak` 直接恢复到稳定阈值 `2`
- 不再重新从 `1/2` 长时间预热

### 什么时候才算真正失锁

只有以下情况才会真正清空稳定态：

- `miss_count > 2`
- 或距离最后一次有效量测超过 `_TRACKER_TIMEOUT_MS`

## 6. ML 推理链路

### 模型输入

模型继续使用训练对齐链路：

- `DataProcessor` 输出逐帧 `MLFeatureData`
- 每个元素带当前帧的 `RD / RA / RE`
- metadata 中会带 `frame_index`

### 模型本体

当前实时模型是 `RACAfallPredictor`：

- `window = 100`
- `stride = 10`

含义：

- 先积满 100 帧再出首个结果
- 之后每 10 帧更新一次新推理结果

### 模型契约

模型仍然经过 `runtime.fall_predictor` 的契约校验：

- 加载模型
- 读取/推断 contract
- 校验当前 `cfg`
- 校验失败则 `ML disabled: cfg mismatch`

这里不会回退规则法。

## 7. ML 告警状态机

### 告警由谁决定

正式告警只由 `MLAlarmState` 决定，不再由规则法决定。

### 告警条件

一次正式 `Fall Alert` 必须同时满足：

1. `monitor_button` 已开启
2. 模型输出 `fall`
3. `probability >= threshold`
4. 连续阳性次数达到 `required_streak`
5. 当前存在稳定目标门控

### tracker_ready 定义

当前 `tracker_ready=True` 的条件是：

- `tracked`
  或
- `predicted` 且 `miss_count <= 2`

并且：

- `stable_latched=True`

### 什么时候会清空 ML 连续阳性

只有这几类情况才会真正 reset：

- `tracker_event == 'lost'`
- `tracker_event == 'relocked'` 且没有恢复稳定锁存
- 模型错误
- contract/cfg mismatch
- 用户关闭监测

不会因为单帧闪断就 reset。

## 8. 前端状态显示

### 首页布局

`real-time` 首页现在分成两段：

- 上方固定区
  - 标题
  - `monitor_button`
  - ML 开关
  - 预设档位
  - 阈值
  - 连续阳性次数
  - 主状态卡
  - 目标门控卡
  - 概率条
  - 确认条
  - 技术摘要
    - 技术摘要区现在是可滚动的只读文本框，长内容不会再被固定高度裁掉
- 下方滚动区
  - 最近推理概率曲线

另外，左侧整个 `real-time` 看板外层已经包了一层 `QScrollArea`，当首屏内容放不下时，可以直接在左侧区域用滚轮向下查看。

### 首页主状态

首页主状态只会显示以下之一：

- `Standby`
- `No Target`
- `Target Warmup`
- `Tracking`
- `Fall Alert`

### 状态含义

- `Standby`
  - 监测未开启
- `No Target`
  - 已监测，但当前没有候选目标
- `Target Warmup`
  - 已锁定，但还没达到稳定门控
- `Tracking`
  - 已经有稳定目标，模型正常运行
- `Fall Alert`
  - ML 正式确认跌倒

### 概率曲线横坐标

概率曲线横坐标现在是：

- `frame index`

不是：

- `inference id`

也不是：

- 时间秒数

## 9. 日志行为

### ML 日志

控制台 `[ML]` 日志已经降频：

- 只在状态变化时输出
- 只在 `positive_pending / alert / error / ready / disabled` 等状态切换时输出
- 不再每次 `non-fall` 都刷一行

### 点云日志

点云健康日志默认关闭。

只有在显式打开环境变量时才会打印：

`RADAR_DEBUG_POINTCLOUD=1`

默认运行时：

- 不会每 50 帧刷 routine 点云日志
- 不会频繁刷“队列大小 / 等待数据”

## 10. 规则法当前定位

规则法相关的高度序列和历史图还保留，但定位已经变成：

- 可视化上下文
- 调试辅助

不再承担：

- 正式跌倒判断
- 正式告警触发

`main.py` 的实时刷新链中也已经删除了旧的规则法死分支，不再保留 `if False` 的实时判定残留。

也就是说，当前真正会触发 `_show_fall_alert()` 的链路只有：

`模型推理 -> MLAlarmState -> Fall Alert`

## 关键函数定位

### 主入口

- `main.py::sendconfigfunc`
- `main.py::openradar`
- `main.py::update_figure`

### 数据与特征

- `real_time_process.py::DataProcessor.run`
- `real_time_process.py::DataProcessor.process_frame_data`
- `runtime/aligned_features.py`

### 跟踪

- `runtime/target_tracking.py::update_tracker`
- `runtime/target_tracking.py::is_tracker_stable`
- `runtime/target_tracking.py::format_tracker_status`

### ML

- `main.py::_consume_ml_feature_frames`
- `runtime/raca_predictor.py`
- `runtime/fall_predictor.py`
- `runtime/ml_alarm.py`

### 前端

- `main.py::_update_realtime_detection_dashboard`
- `main.py::_refresh_monitor_status_preview`
- `main.py::_set_monitor_enabled`
