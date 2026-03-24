# IWR6843 实时跌倒检测与可视化系统

## 项目概览

这是一个基于 TI IWR6843 毫米波雷达的实时跌倒检测项目。当前版本已经围绕 `config/Radar.cfg`、实时 DSP 处理链、单目标跟踪、规则法跌倒检测、模型加载接口和界面可视化做了统一整理，目标是让项目具备以下能力：

- 以 `Radar.cfg` 作为运行时唯一配置真源
- 实时接收雷达数据并生成多种热图
- 从点云中锁定单个主目标并构建高度历史
- 使用规则法进行实时跌倒检测
- 预留后续深度学习模型加载接口
- 预留离线 `.bin` 数据回放接口

当前仓库的重点已经从早期的手势识别/采样实验逻辑，收敛为“实时跌倒检测 + 模型接口预留 + 回放接口预留”。

## 当前版本已经实现了什么

### 1. `Radar.cfg` 运行时配置化

项目启动后会解析当前选中的 cfg 文件，默认使用：

```text
config/Radar.cfg
```

并把解析结果作为统一的运行时配置，下发到：

- cfg 信息显示
- UDP 采集链
- 实时数据处理线程
- DSP 处理链
- 热图尺寸和相关参数

这意味着当前系统不再依赖硬编码的 `ADC sample / chirp / TX / RX / frame_length`。

### 2. 实时 DSP 处理链

实时处理链已经可以稳定生成并显示以下特征：

- 时间-距离图 `RTI`
- 多普勒-时间图 `DTI`
- 距离-多普勒图 `RDI`
- 距离-方位角图 `RAI`
- 距离-俯仰角图 `REI`

同时会从角度图中提取点云，用于后续目标聚类、目标跟踪和跌倒检测。

### 3. 单目标跟踪

当前版本已不再直接拿聚类结果中的第一个目标做跌倒检测，而是新增了独立的单目标跟踪模块：

- 首次锁定
- 连续帧关联
- 丢失目标
- 重锁新目标

当发生失锁或重锁时，会自动清空高度历史并重置状态，避免把两段不连续目标误拼成一次跌倒事件。

### 4. 规则法跌倒检测

跌倒检测算法已经从 `main.py` 中独立到 `runtime/fall_detection.py`，目前只保留一套新策略规则法，不再区分旧/新两套入口。

当前规则法大致基于：

- 高度变化
- 下降速度
- 低姿态持续时间
- 运动过程中的空间变化

来判断是否发生跌倒。

### 5. 模型加载接口

项目已经预留后续模型接入能力，当前支持：

- `.py` 预测器插件
- `TorchScript` 模型文件：`.pt` / `.jit` / `.ts`

说明：

- 当前模型推理结果是“并行辅助结果”
- 正式跌倒告警仍由规则法决定
- 模型不会直接接管报警逻辑

这套设计是为了方便你后续离线训练模型后，再把模型文件加载到实时界面中做在线推理。

### 6. 数据回放接口预留

界面中已经预留 `.bin` 回放入口，但当前版本只是“接口预留”，尚未把：

```text
.bin -> 原始帧解析 -> DSP -> UI 实时热图刷新
```

这条离线回放链完全接通。

也就是说：

- 可以加载并登记回放文件入口
- 但当前还没有实现真正的 `.bin` 实时播放

## 当前版本没有实现，或仅做了接口预留的内容

以下内容当前不是完整功能：

- 不实现新的采集/抓取链
- 不再保留早期“通用采样 / event snapshot / dataset record”运行链
- 不实现真正的 `.bin` 离线回放渲染
- 不默认支持直接加载 `.pth` 权重并自动推理

其中 `.pth` 之所以没有直接支持，是因为它通常只包含参数，不包含模型结构；如果你后续需要加载 `.pth`，建议提供一个 `.py` 插件包装器，把结构和权重加载逻辑写进去，再通过当前接口接入。

## 当前项目的整体链条

### 1. 配置链

1. 选择 `config/Radar.cfg`
2. 解析 cfg
3. 生成统一 `runtime_cfg`
4. 同步给 UI、采集线程和 DSP

### 2. 实时数据链

1. 通过串口发送 cfg 到雷达
2. UDP 监听线程接收原始数据
3. 数据处理线程完成原始帧重组
4. 交给 DSP 生成 `RTI / DTI / RDI / RAI / REI`
5. 从角度图中提取点云
6. 主界面刷新热图和点云相关显示

### 3. 跌倒检测链

1. 对点云做聚类，生成候选目标中心
2. 交给单目标跟踪器锁定主目标
3. 从主目标提取高度信息，更新 `height_history`
4. `fall_detection.detect_fall()` 做规则法判定
5. 如果检测到跌倒，则更新状态面板并触发报警显示

### 4. 模型推理链

1. 实时热图和高度历史会被打包成特征片段
2. 如果用户加载了模型，系统会调用模型接口做并行推理
3. 推理结果显示在界面中
4. 模型结果当前仅作辅助显示，不接管规则法报警

### 5. 回放链

1. 用户可在界面中选择 `.bin` 文件
2. 系统登记该回放文件
3. 当前版本仅完成接口预留，尚未完成真实回放

## 核心模块说明

### `main.py`

主界面和业务编排入口，负责：

- 启动 Qt 界面
- 解析和应用 `Radar.cfg`
- 发送 cfg 到雷达
- 启动实时线程
- 刷新热图
- 调用单目标跟踪器
- 调用跌倒检测
- 调用模型接口
- 更新状态面板

### `real_time_process.py`

负责实时数据链路：

- UDP 数据监听
- 原始帧重组
- 数据处理线程调度

### `DSP.py`

负责雷达 DSP 主处理链：

- 距离向处理
- 多普勒向处理
- 角度图生成
- 点云提取

### `runtime/fall_detection.py`

负责纯算法层的跌倒检测逻辑：

- 高度历史更新
- 异常点过滤
- 规则法跌倒判定

### `runtime/target_tracking.py`

负责单目标跟踪：

- 锁定主目标
- 跟踪候选目标
- 丢失与重锁判断

### `runtime/fall_predictor.py`

负责后续机器学习/深度学习模型接口：

- 统一预测器接口
- 加载 `.py` 插件
- 加载 `TorchScript`
- 返回统一预测结果

### `runtime/replay_controller.py`

负责回放入口管理：

- 选择 `.bin`
- 校验回放源
- 预留后续回放控制接口

### `runtime/radar_config.py`

负责在线运行时与雷达/DCA1000 的控制接口：

- 串口发送 cfg
- `sensorStart / sensorStop`
- DCA1000 基础配置命令

### `UI_interface.py`

Qt Designer 生成的界面定义文件，当前界面已经围绕：

- 实时热图显示
- 跌倒状态显示
- 模型推理入口
- 数据回放入口

进行了整理，旧的采样链控件已经清理出运行主路径。

### `iwr6843_tlv/detected_points.py`

负责 cfg 解析与运行参数提取，是当前 `Radar.cfg` 配置链的重要入口之一。

## 当前界面如何使用

### 1. 启动程序

```powershell
python main.py
```

### 2. 选择并发送 cfg

右侧配置区默认使用：

```text
config/Radar.cfg
```

确认串口正确后点击 `send`。

### 3. 开始实时显示

发送成功后，系统会启动：

- UDP 监听线程
- 数据处理线程

随后五张热图会开始实时刷新。

### 4. 开启跌倒监测

当前跌倒监测相关控制主要分两处：

- 右侧 `开始监测 / 停止监测` 按钮：控制状态面板进入监测状态
- 高度检测页中的“启用跌倒检测”：控制规则法是否真正参与报警判定

因此如果你已经启动雷达但右侧还是 `Standby`，通常需要确认：

- 是否点击了“开始监测”
- 是否在高度检测页启用了跌倒检测

### 5. 加载模型

如果你已经离线训练好了模型，可在右侧模型推理区域加载：

- `.py`
- `.pt`
- `.jit`
- `.ts`

加载成功后，模型预测结果会在实时运行中作为辅助信息显示。

### 6. 预留回放入口

右侧“数据回放”区域目前可以选择 `.bin` 文件，但还没有真正播放能力。

## 模型文件接口说明

### 推荐方式 1：TorchScript

如果你后续离线训练后能导出 TorchScript，推荐使用：

- `.pt`
- `.jit`
- `.ts`

这样可以直接通过当前界面加载。

### 推荐方式 2：`.py` 插件

如果你的模型需要自定义预处理、模型结构或权重加载逻辑，可以写一个 `.py` 插件，通过项目现有接口加载。

### 不建议直接裸加载 `.pth`

`.pth` 往往只保存参数，不包含结构定义。当前项目不会直接猜测模型结构，因此更适合：

- 先导出 TorchScript
- 或提供 `.py` 包装器

## 面向离线 `.bin` 训练模型的操作手册

这一节面向你后续的真实工作流：

```text
离线采集 .bin -> 离线特征提取/训练 -> 导出模型 -> 挂回当前实时窗口
```

这里重点不是重新实现采集，而是说明当前项目已经预留了哪些接口，以及你离线训练时怎样和现有实时系统保持一致。

### 一、先明确当前项目里的边界

当前仓库已经具备：

- `Radar.cfg` 运行时真源
- 实时 DSP 热图与点云处理链
- 单目标跟踪
- 规则法跌倒检测
- 模型加载接口
- `.bin` 回放入口预留

当前仓库还没有完整具备：

- 从 `.bin` 到 DSP 的正式离线回放链
- 标准化的离线特征提取脚本
- 完整训练脚本
- 训练后自动部署脚本

所以推荐的工作方式是：

1. 你在离线流程里自己处理 `.bin`
2. 但特征格式要尽量对齐当前实时系统
3. 最后把模型导出成当前系统能直接加载的形式

### 二、离线训练时必须对齐的配置

离线训练和实时部署必须使用同一套雷达配置，当前推荐固定使用：

```text
config/Radar.cfg
```

原因很直接：

- `ADC samples`
- `range bins`
- `doppler bins`
- `frame_length`
- 距离/速度分辨率
- 角度图尺寸

都会影响最终特征张量的 shape。

如果你离线训练时用的是另一份 cfg，而实时窗口运行时用的是 `Radar.cfg`，那很容易出现：

- 模型输入 shape 不一致
- 归一化统计失效
- 特征物理含义偏移
- 模型上线后效果明显变差

因此建议：

- 训练集版本和 cfg 强绑定
- 每一版模型都记录对应的 `Radar.cfg`
- 如果以后改 cfg，就直接视为一个新数据版本和新模型版本

### 三、推荐的离线训练链条

推荐你按下面这条线组织离线流程：

1. 读取 `.bin`
2. 按 `Radar.cfg` 还原原始雷达帧
3. 用与当前项目一致的 DSP 逻辑生成特征
4. 组装成时间片段 clip
5. 做标签和数据集划分
6. 训练模型
7. 导出可部署模型
8. 在当前 GUI 中加载

也就是说，离线训练时最重要的不是“重新定义一套新特征”，而是尽量复用当前项目已经稳定下来的特征语义。

### 四、建议对齐当前实时系统的特征

当前实时系统在模型接口里预留的 clip 结构是 `FallFeatureClip`，字段如下：

```python
FallFeatureClip(
    timestamp_start_ms=...,
    timestamp_end_ms=...,
    RT=...,
    DT=...,
    RDT=...,
    ART=...,
    ERT=...,
    height_history=...,
    tracked_target_state=...,
    runtime_cfg=...,
)
```

因此你离线训练时，建议尽量围绕以下输入组织数据：

- `RT`
  时间-距离特征
- `DT`
  多普勒-时间特征
- `RDT`
  距离-多普勒特征
- `ART`
  距离-方位角特征
- `ERT`
  距离-俯仰角特征
- `height_history`
  主目标高度历史
- `tracked_target_state`
  主目标当前状态摘要
- `runtime_cfg`
  当前配置摘要

这里的推荐原则是：

- 如果你的模型只想先吃一部分模态，也没问题
- 但离线特征命名和语义最好与当前接口一致
- 后续挂回实时窗口时会更顺

### 五、推荐的 clip 组织方式

当前实时接口是 clip 级的，所以离线训练时也建议不要做“单帧分类”，而是做“时间片段分类”。

推荐思路：

- 每个样本是一个 clip，而不是单帧
- clip 内包含连续多帧特征
- clip 时长建议覆盖完整动作过程

例如可以按下面方式组织：

- `clip_length = 2s ~ 4s`
- `stride = 0.5s ~ 1s`
- 标签以 clip 为单位

如果你要训“跌倒 vs 非跌倒”，可以把完整跌倒过程切成多个重叠 clip；如果你要训多分类，也可以扩展成：

- `fall`
- `near_fall`
- `sit_down`
- `lie_down`
- `stand`
- `walk`
- `other`

### 六、建议的数据目录与元信息

当前仓库没有强制你必须使用哪种离线目录结构，但为了后续维护清晰，建议至少做到：

```text
dataset/
├─ subject_01/
│  ├─ session_01/
│  │  ├─ fall/
│  │  ├─ sit_down/
│  │  └─ walk/
│  └─ session_02/
└─ subject_02/
```

每个 clip 建议保存：

- 特征数组
- 标签
- 来源 `.bin`
- 对应 `Radar.cfg`
- 采样时间范围
- 受试者 ID
- 场次 ID

最少建议保留一份元信息，例如：

```json
{
  "source_bin": "xxx.bin",
  "label": "fall",
  "subject_id": "subject_01",
  "session_id": "session_01",
  "cfg": "Radar.cfg",
  "frame_start": 120,
  "frame_end": 200
}
```

### 七、训练阶段建议

你后续离线训练时，建议遵循这几个原则：

- 划分训练/验证/测试集时，尽量按 `session` 或 `subject` 划分
- 不要随机把同一段动作的相邻 clip 同时分进 train 和 test
- 先训练一个简单基线模型，再逐步增加模态
- 先保证输入契约稳定，再追求更复杂网络

推荐的起步方式：

1. 先只用 `RT + DT + height_history`
2. 跑一个简单 2D CNN / CNN+MLP 基线
3. 确认标签和样本划分合理
4. 再逐步加入 `RDT / ART / ERT`

这样做的好处是：

- 更容易排查标签和数据问题
- 更容易确认到底是模型问题还是特征问题
- 更适合和当前规则法结果做对照

### 八、导出模型时的建议

当前实时系统最适合接的模型产物有两种。

#### 方案 A：导出 TorchScript

这是最推荐的方式。

你训练完后，把模型导出成：

- `.pt`
- `.jit`
- `.ts`

然后在当前 GUI 右侧“模型推理”区域直接加载。

优点：

- 对当前系统最直接
- 部署路径最短
- 不需要在实时工程里再写复杂加载逻辑

#### 方案 B：写一个 `.py` 预测器插件

如果你的模型除了权重，还依赖：

- 特殊预处理
- 多模态拼接
- 特定归一化
- 自定义后处理

那么更推荐你写一个 `.py` 插件。

插件至少需要返回一个 `BaseFallPredictor` 实例，可以通过以下任一方式暴露：

- `build_fall_predictor()`
- `build_predictor()`
- `PREDICTOR`

一个最小插件示意如下：

```python
from runtime.fall_predictor import BaseFallPredictor, FallPrediction


class MyPredictor(BaseFallPredictor):
    def predict(self, clip):
        return FallPrediction(
            available=True,
            label="fall",
            score=0.95,
            probability=0.91,
            topk=[("fall", 0.91), ("non-fall", 0.09)],
            metadata={"source": "custom-plugin"},
        )

    def reset(self):
        pass


def build_fall_predictor():
    return MyPredictor()
```

如果你的离线模型最终只有 `.pth`：

- 不建议直接在当前 GUI 里裸加载 `.pth`
- 更适合在插件里自己定义网络结构并加载权重

### 九、怎样把离线模型挂回当前实时窗口

你后续把模型挂回当前项目时，可以按下面流程操作：

1. 确保实时系统仍然使用训练时对应的 `Radar.cfg`
2. 启动 `main.py`
3. 正常发送 cfg，启动实时链
4. 在右侧“模型推理”区域选择模型文件
5. 点击 `load model`
6. 让实时系统进入监测状态
7. 观察状态面板里的模型辅助结果

这里要特别注意：

- 模型推理目前是辅助信息
- 正式跌倒告警仍然由规则法触发
- 所以如果模型判断和规则法不同，界面上会反映模型结果，但不会直接替代规则法报警

这是一种比较稳妥的接入方式，适合你先做离线验证，再逐步观察在线表现。

### 十、建议的联调顺序

后续真正接模型时，建议你按这个顺序推进：

1. 先确认离线训练数据和 `Radar.cfg` 完全一致
2. 先导出一个最简单可运行的 TorchScript 或插件
3. 先在实时系统中确认“模型能加载”
4. 再确认“模型能推理”
5. 再观察实时窗口中的预测标签是否稳定
6. 最后再考虑是否让模型参与更强的业务决策

不要一开始就把模型结果直接接管报警，这样更容易定位问题。

### 十一、当前最推荐的工程分工

如果你后续继续推进，我推荐把工程拆成两条明确的线：

- 当前仓库：实时系统、规则法、模型加载、在线显示
- 离线训练工程：`.bin` 解析、特征生成、数据集构建、训练、导出模型

两边通过这几个东西对齐：

- 同一份 `Radar.cfg`
- 同一套特征语义
- 同一套标签定义
- 同一套模型输入契约

这样仓库职责会很清楚，也更利于后续维护。

### 十二、当前仓库里的离线特征提取入口

当前仓库已经提供了一个和实时 DSP 链对齐的离线提取器：

```text
offline/feature_extractor.py
```

它当前的设计目标是：

- 复用 `Radar.cfg` 解析
- 复用实时链同构的原始帧解码方式
- 复用 `DSP.RDA_Time()` 和 `DSP.Range_Angle()`
- 把 `.bin` 提取成事件级训练特征，而不是旧版那种逐帧增长 shape 的缓存产物

#### 默认输出

每个事件样本输出到：

```text
out_root/<label>/<clip_id>/
├─ RT.npy
├─ DT.npy
├─ RDT.npy
├─ ART.npy
├─ ERT.npy
└─ meta.json
```

其中当前默认特征 shape 为：

- `RT`: `(T, 4, 128)`
- `DT`: `(T, 64)`
- `RDT`: `(T, 128, 64)`
- `ART`: `(T, 361, 128)`
- `ERT`: `(T, 361, 128)`

在你当前采集配置下，整段 `.bin` 默认是：

- `T = 100`
- `5 秒`
- `20 fps`

#### 推荐用法：manifest 批处理

推荐用一个外部 `JSONL` 清单驱动批处理：

```powershell
python -m offline.feature_extractor `
  --manifest samples.jsonl `
  --cfg f:\test\cfg\Radar.cfg `
  --out features
```

`JSONL` 每行一个样本，最少包含：

```json
{"bin_path":"F:\\Data_bin\\...\\xxx.bin","label":"fall"}
```

可选字段包括：

- `clip_id`
- `subject_id`
- `session_id`
- `scene`
- `frame_start`
- `frame_end`
- `metadata`

一个更完整的例子：

```json
{
  "bin_path": "F:\\Data_bin\\dormitory\\walk\\walk_S01_dormitory_02_Raw_0.bin",
  "label": "walk",
  "clip_id": "walk_S01_dormitory_02",
  "subject_id": "S01",
  "session_id": "dormitory_02",
  "scene": "dormitory",
  "metadata": {
    "class_names": ["fall", "walk", "sit_down"],
    "positive_labels": ["fall"]
  }
}
```

#### 单文件调试模式

如果你只是想先看某个 `.bin` 能不能正常抽特征，也可以直接对单个文件运行：

```powershell
python -m offline.feature_extractor `
  --bin F:\Data_bin\dormitory\walk\walk_S01_dormitory_02_Raw_0.bin `
  --cfg f:\test\cfg\Radar.cfg `
  --out debug_features `
  --label walk
```

#### 当前实现边界

这个提取器当前只负责：

- `.bin -> 事件级特征`
- `manifest -> 批量事件目录`
- `meta.json -> 训练所需元信息`

它当前不负责：

- 训练脚本本身
- `.bin` 回放到实时 GUI
- 自动把训练好的模型导出为 TorchScript
- 直接接管在线跌倒告警逻辑

因此它更适合作为“离线训练前的数据准备层”。

### 十三、当前仓库里的训练辅助入口

为了避免训练相关脚本继续堆在根目录里，当前仓库已经新增：

```text
training/
```

目前包含两个实用入口：

- `training/build_manifest.py`
  从已经提取好的事件特征目录扫描生成训练用 `JSONL` manifest
- `training/export_model.py`
  把训练产物整理成“模型文件 + model_meta.json”的可部署目录

#### 1. 从特征目录生成训练 manifest

```powershell
python -m training.build_manifest `
  --features-root features `
  --out training_manifest.jsonl
```

这个 manifest 的输入是已经提取好的事件目录，而不是原始 `.bin`。

#### 2. 打包训练好的模型

```powershell
python -m training.export_model `
  --model model.pt `
  --out packaged_model `
  --cfg f:\test\cfg\Radar.cfg `
  --sample-meta features\fall\clip_001\meta.json `
  --class-names fall walk sit_down `
  --positive-labels fall
```

这个步骤当前主要负责：

- 拷贝模型文件
- 生成 `model_meta.json`
- 固化 `cfg_sha256`
- 固化 `clip_frames / frame_periodicity_ms / feature_shapes`

这样你后面把模型挂回实时窗口时，训练侧和部署侧就有了一份稳定的元信息契约。

## 当前测试情况

项目当前已经包含以下方向的测试：

- `Radar.cfg` 运行时解析
- DSP/runtime shape 一致性
- 跌倒检测规则法
- 单目标跟踪
- 模型接口占位
- 回放控制器占位

运行方式：

```powershell
python -m unittest discover -s tests -v
```

## 依赖环境

建议使用 Python 3.10 左右环境。

主要依赖见 `requirements.txt`，当前包括：

- `PyQt5`
- `pyqtgraph`
- `numpy`
- `torch`
- `matplotlib`
- `pyserial`
- `scikit-learn`
- `joblib`

安装示例：

```powershell
pip install -r requirements.txt
```

## 当前目录中的关键文件

```text
.
├─ config/
│  └─ Radar.cfg
├─ dsp/
├─ iwr6843_tlv/
├─ tests/
├─ DSP.py
├─ offline/
│  ├─ __init__.py
│  └─ feature_extractor.py
├─ runtime/
│  ├─ __init__.py
│  ├─ fall_detection.py
│  ├─ fall_predictor.py
│  ├─ radar_config.py
│  ├─ replay_controller.py
│  └─ target_tracking.py
├─ support/
│  ├─ __init__.py
│  ├─ colortrans.py
│  ├─ globalvar.py
│  └─ pointcloud_clustering.py
├─ training/
│  ├─ __init__.py
│  ├─ build_manifest.py
│  └─ export_model.py
├─ main.py
├─ real_time_process.py
└─ UI_interface.py
```

## 当前版本的限制

- 当前跌倒检测仍以规则法为正式报警主链
- 模型推理是辅助结果，不直接接管告警
- 回放接口只做了入口预留，未完成真实 `.bin` 播放
- 当前主目标跟踪按单人优先设计，不保证多人场景的稳定性
- 跌倒监测逻辑仍依赖“高度检测页启用跌倒检测”这一业务开关

## 后续推荐方向

如果后续继续沿当前项目推进，最自然的下一步是：

1. 用 `Radar.cfg` 和离线 `.bin` 数据打通真正的回放链
2. 完成离线特征提取与训练脚本
3. 把训练好的模型导出为 TorchScript 或 `.py` 插件
4. 接入当前实时模型接口
5. 在规则法之外增加模型辅助判断，逐步验证是否需要让模型参与最终报警策略

## 总结

当前项目已经完成从“早期实验型雷达可视化工程”向“面向跌倒检测的实时系统”收敛的主要工作：

- 配置链统一
- DSP 链统一
- 跌倒检测模块化
- 单目标跟踪独立化
- 模型接口预留
- 回放接口预留
- 旧采样链与旧手势语义基本清理

如果你的下一步是“基于离线 `.bin` 数据训练模型，再把模型挂回实时窗口”，当前这版项目结构已经可以作为稳定基础继续往下扩展。
