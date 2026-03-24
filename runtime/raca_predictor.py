"""RACA v2 实时跌倒检测推理器.

将训练好的 RACA_v2 模型接入 runtime.fall_predictor 框架，
实现基于 RD/RA/RE 特征帧的实时跌倒分类。

使用方式 (在 main.py 中)::

    from runtime.raca_predictor import load_raca_predictor
    predictor = load_raca_predictor(
        checkpoint='checkpoints/transformer_20260317/raca_v2_best.pth',
        device='cpu',
    )

挂载到 fall_predictor 全局::

    import runtime.fall_predictor as fp
    fp._GLOBAL_PREDICTOR = predictor
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from runtime.fall_predictor import (
    BaseFallPredictor,
    FallFeatureClip,
    FallPrediction,
    PredictorLoadResult,
)


# ---------------------------------------------------------------------------
# 常量 (与 train_v2.py 保持一致)
# ---------------------------------------------------------------------------
T_FRAMES  = 100   # 每个 clip 的帧数
R_BINS    = 128   # range bins
D_BINS    = 64    # doppler bins
A_BINS    = 361   # angle bins


# ---------------------------------------------------------------------------
# 帧缓冲区：滑动窗口积累 T_FRAMES 帧
# ---------------------------------------------------------------------------
class FrameBuffer:
    """滑动窗口缓冲区，积累 T_FRAMES 帧后触发推理。

    Args
    ----
    window   : 窗口大小 (帧数)
    stride   : 每隔多少帧推理一次 (stride=1 每帧都推理, stride=window 不重叠)
    """

    def __init__(self, window: int = T_FRAMES, stride: int = 10):
        self.window = window
        self.stride = stride
        self._rd: Deque[np.ndarray] = deque(maxlen=window)  # 每帧 (R, D)
        self._ra: Deque[np.ndarray] = deque(maxlen=window)  # 每帧 (A, R)
        self._re: Deque[np.ndarray] = deque(maxlen=window)  # 每帧 (A, R)
        self._count = 0  # 累计帧数

    def push(self, rd_frame: np.ndarray,
             ra_frame: np.ndarray,
             re_frame: np.ndarray) -> bool:
        """压入一帧，返回 True 表示缓冲区已满且到达 stride 触发点。"""
        self._rd.append(rd_frame.astype(np.float32))
        self._ra.append(ra_frame.astype(np.float32))
        self._re.append(re_frame.astype(np.float32))
        self._count += 1
        ready = (len(self._rd) == self.window) and (self._count % self.stride == 0)
        return ready

    def get_clip(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回当前窗口的 (rd, ra, re)，shape 分别为 (T,R,D), (T,A,R), (T,A,R)."""
        return (
            np.stack(self._rd),   # (T, R, D)
            np.stack(self._ra),   # (T, A, R)
            np.stack(self._re),   # (T, A, R)
        )

    def reset(self):
        self._rd.clear()
        self._ra.clear()
        self._re.clear()
        self._count = 0

    @property
    def current_size(self) -> int:
        return len(self._rd)


# ---------------------------------------------------------------------------
# RACA 推理器
# ---------------------------------------------------------------------------
class RACAfallPredictor(BaseFallPredictor):
    """基于 RACA_v2 模型的实时跌倒推理器.

    Args
    ----
    model      : 已加载的 RACA_v2 模型实例
    label_map  : {类别名: 索引} 字典
    class_names: 类别名列表 (按索引排列)
    device     : 推理设备
    window     : 滑动窗口帧数 (默认 100)
    stride     : 推理步长 (默认 10，即每10帧推理一次)
    fall_label : 跌倒类别名 (默认 'fall')
    threshold  : 跌倒概率阈值 (默认 0.5)
    """

    def __init__(
        self,
        model: Any,
        label_map: Dict[str, int],
        class_names: List[str],
        device: str = "cpu",
        window: int = T_FRAMES,
        stride: int = 10,
        fall_label: str = "fall",
        threshold: float = 0.5,
        binary_mode: bool = False,
    ):
        import torch
        self._model = model
        self._model.eval()
        self._model.to(device)
        self._device = device
        self._label_map = label_map
        self._class_names = class_names
        self._fall_idx = label_map.get(fall_label, 0)
        self._threshold = threshold
        self._binary_mode = binary_mode
        self._buffer = FrameBuffer(window=window, stride=stride)
        self._last_prediction: Optional[FallPrediction] = None
        self._inference_id = 0
        self._torch = torch

    # ------------------------------------------------------------------
    # BaseFallPredictor 接口
    # ------------------------------------------------------------------
    def predict(self, clip: FallFeatureClip) -> FallPrediction:
        """主接口: 接收一个 FallFeatureClip，返回预测结果.

        clip.RDT : 当前帧的 RD 图谱  (range, doppler)  np.ndarray
        clip.ART : 当前帧的 RA 图谱  (angle, range)    np.ndarray
        clip.ERT : 当前帧的 RE 图谱  (angle, range)    np.ndarray

        若帧尚未积累足够，返回上次结果（或 unavailable）。
        """
        # 提取当帧特征
        rd_frame = self._extract_frame(clip.RDT, (R_BINS, D_BINS))
        ra_frame = self._extract_frame(clip.ART, (A_BINS, R_BINS))
        re_frame = self._extract_frame(clip.ERT, (A_BINS, R_BINS))

        ready = self._buffer.push(rd_frame, ra_frame, re_frame)
        buffer_metadata = {
            "buffered_frames": self._buffer.current_size,
            "required_frames": self._buffer.window,
        }

        if not ready:
            # 缓冲区未满，返回上次结果
            if self._last_prediction is not None:
                metadata = dict(self._last_prediction.metadata or {})
                metadata.update(buffer_metadata)
                return FallPrediction(
                    available=self._last_prediction.available,
                    label=self._last_prediction.label,
                    score=self._last_prediction.score,
                    probability=self._last_prediction.probability,
                    topk=list(self._last_prediction.topk),
                    metadata=metadata,
                )
            return FallPrediction(
                available=False,
                metadata={
                    "reason": "accumulating frames",
                    **buffer_metadata,
                },
            )

        # 运行推理
        self._last_prediction = self._run_inference()
        metadata = dict(self._last_prediction.metadata or {})
        metadata.update(buffer_metadata)
        self._last_prediction.metadata = metadata
        return self._last_prediction

    def reset(self) -> None:
        """跟踪目标丢失时清空缓冲区."""
        self._buffer.reset()
        self._last_prediction = None
        self._inference_id = 0

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _coerce_feature_to_2d(
        self,
        arr: np.ndarray,
        expected_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Reduce live queue tensors to a single 2D frame before resizing."""
        arr = np.asarray(arr, dtype=np.float32)

        while arr.ndim > 2:
            if arr.ndim >= 3 and arr.shape[-2:] == expected_shape:
                arr = arr[-1]
                continue
            if arr.ndim == 3 and arr.shape[:2] == expected_shape:
                arr = arr.mean(axis=-1)
                continue
            if arr.shape[0] <= 32:
                arr = arr[-1]
                continue
            if arr.shape[-1] <= 32:
                arr = arr.mean(axis=-1)
                continue
            arr = arr.mean(axis=0)

        return arr.astype(np.float32, copy=False)

    def _extract_frame(self, arr: Any, expected_shape: Tuple[int, int]) -> np.ndarray:
        """将任意输入规整为 expected_shape 的 float32 数组."""
        if arr is None:
            return np.zeros(expected_shape, dtype=np.float32)
        arr = self._coerce_feature_to_2d(arr, expected_shape)
        if arr.shape == expected_shape:
            return arr
        # 尺寸不匹配时自适应 resize
        try:
            from skimage.transform import resize
            return resize(arr, expected_shape, anti_aliasing=True).astype(np.float32)
        except ImportError:
            # fallback: 截断或填充
            out = np.zeros(expected_shape, dtype=np.float32)
            r = min(arr.shape[0], expected_shape[0])
            c = min(arr.shape[1], expected_shape[1])
            out[:r, :c] = arr[:r, :c]
            return out

    def _run_inference(self) -> FallPrediction:
        """从缓冲区取 clip，运行 RACA_v2 前向传播."""
        torch = self._torch
        rd, ra, re = self._buffer.get_clip()
        self._inference_id += 1

        # 增加 batch 维度: (1, T, R, D), (1, T, A, R), (1, T, A, R)
        rd_t = torch.from_numpy(rd).unsqueeze(0).to(self._device)
        ra_t = torch.from_numpy(ra).unsqueeze(0).to(self._device)
        re_t = torch.from_numpy(re).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(rd_t, ra_t, re_t)   # (1, 1) for binary, (1, C) for multi-class

        if self._binary_mode:
            # Binary: 1 logit -> sigmoid -> fall probability
            fall_prob = float(torch.sigmoid(logits).item())
            pred_label = "fall" if fall_prob >= self._threshold else "non-fall"
            topk = sorted(
                [("fall", fall_prob), ("non-fall", 1.0 - fall_prob)],
                key=lambda x: x[1],
                reverse=True,
            )
            raw_score = float(logits.item())
        else:
            # Multi-class: softmax
            probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
            fall_prob = float(probs[self._fall_idx])
            pred_idx  = int(np.argmax(probs))
            pred_label = self._class_names[pred_idx]
            topk = sorted(
                [(self._class_names[i], float(probs[i])) for i in range(len(probs))],
                key=lambda x: x[1], reverse=True
            )
            raw_score = float(logits.cpu().numpy().reshape(-1)[pred_idx])

        return FallPrediction(
            available=True,
            label=pred_label,
            score=raw_score,
            probability=fall_prob,
            topk=topk,
            metadata={
                "device": self._device,
                "threshold": self._threshold,
                "binary_mode": self._binary_mode,
                "inference_id": self._inference_id,
                "buffered_frames": self._buffer.current_size,
                "required_frames": self._buffer.window,
            },
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def load_raca_predictor(
    checkpoint: str,
    device: str = "cpu",
    window: int = T_FRAMES,
    stride: int = 10,
    fall_label: str = "fall",
    threshold: float = 0.5,
) -> PredictorLoadResult:
    """从 checkpoint 文件加载 RACA_v2 推理器.

    Args
    ----
    checkpoint : raca_v2_best.pth 路径
    device     : 'cpu' 或 'cuda'
    window     : 滑动窗口帧数
    stride     : 推理步长 (帧)
    fall_label : 跌倒类别名
    threshold  : 跌倒概率阈值

    Returns
    -------
    PredictorLoadResult
    """
    try:
        import torch
        import sys, os
        # 确保能找到 training.train_v2
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from training.train_v2 import RACA_v2

        ckpt = torch.load(checkpoint, map_location=device)
        label_map   = ckpt["label_map"]          # {label_str: idx}
        class_names = ckpt["class_names"]        # [label_str, ...]
        args        = ckpt.get("args", {})
        binary_mode = ckpt.get("binary_mode", False)
        re_weight   = ckpt.get("re_weight", args.get("re_weight", 1.0))
        # binary_mode 时模型输出维度为 1，多分类为 len(class_names)
        actual_classes = 1 if binary_mode else len(class_names)

        model = RACA_v2(
            num_classes=actual_classes,
            embed_dim=args.get("embed_dim", 64),
            temporal_type=ckpt["cfg"]["temporal_type"],
            temporal_cfg=ckpt["cfg"]["temporal_cfg"],
            dropout=0.0,   # 推理时关闭 dropout
            angle_reduced=args.get("angle_reduced", 64),
            re_weight=re_weight,
        )
        model.load_state_dict(ckpt["model"])

        predictor = RACAfallPredictor(
            model=model,
            label_map=label_map,
            class_names=class_names,
            device=device,
            window=window,
            stride=stride,
            fall_label=fall_label,
            threshold=threshold,
            binary_mode=binary_mode,
        )

        return PredictorLoadResult(
            success=True,
            predictor=predictor,
            message=f"RACA_v2 loaded: {len(class_names)} classes, "
                    f"val_acc={ckpt.get('val_acc', 0):.1f}%",
            model_path=checkpoint,
        )

    except Exception as exc:
        from runtime.fall_predictor import NullFallPredictor
        return PredictorLoadResult(
            success=False,
            predictor=NullFallPredictor(),
            message=f"Failed to load RACA_v2: {exc}",
            model_path=checkpoint,
        )
