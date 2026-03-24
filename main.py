# ========== 必须在所有导入之前设置环境变量 ==========
import os
import multiprocessing
from collections import deque
# 修复 Windows 上 joblib/loky 检测物理 CPU 核心数失败的问题
# 设置 LOKY_MAX_CPU_COUNT 为逻辑核心数，避免尝试检测物理核心数
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
# 注意：Windows 上 multiprocessing 不支持 'threading' 上下文，已移除 JOBLIB_START_METHOD 设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ====================================================

import DSP
from runtime import fall_detection as fd
from runtime import ml_alarm as mla
from runtime import fall_predictor as fp
from runtime import replay_controller as rc
import real_time_process as rtp
from runtime import target_tracking as tt
from real_time_process import UdpListener, DataProcessor
from runtime.radar_config import SerialConfig
from runtime.radar_config import DCA1000Config
from queue import Empty, Queue

_TORCH_PRELOAD_OK, _TORCH_PRELOAD_MESSAGE = fp.preload_torch_runtime()

import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem, GLGridItem, GLAxisItem
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import time
import sys
import numpy as np
from serial.tools import list_ports
import iwr6843_tlv.detected_points as readpoint
from support import globalvar as gl
from support import pointcloud_clustering

# 在导入 pointcloud_clustering 后，尝试 patch joblib/loky（如果 pointcloud_clustering 已经导入了 sklearn）
# 这样可以确保即使 pointcloud_clustering 中导入了 sklearn，我们也能 patch
try:
    import joblib.externals.loky.backend.context as loky_context
    if hasattr(loky_context, '_count_physical_cores'):
        def _patched_count_physical_cores():
            """Patch 后的函数，直接返回逻辑核心数，避免调用 wmic"""
            return multiprocessing.cpu_count()
        loky_context._count_physical_cores = _patched_count_physical_cores
except (ImportError, AttributeError):
    # 如果模块不存在或没有该函数，忽略
    pass

import warnings
# 过滤 NumPy 版本警告（scikit-learn 发出的）
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy version.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*')
# 过滤 joblib/loky 警告（更全面的过滤）
warnings.filterwarnings('ignore', message='.*LOKY_MAX_CPU_COUNT.*')
warnings.filterwarnings('ignore', message='.*Returning the number of logical cores.*')
warnings.filterwarnings('ignore', message='.*系统找不到指定的文件.*')
warnings.filterwarnings('ignore', message='.*WinError 2.*')
warnings.filterwarnings('ignore', message='.*_count_physical_cores.*')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')
# 过滤所有 UserWarning 中与 CPU 核心数相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib.*')
warnings.filterwarnings('ignore', category=UserWarning, module='loky.*')

import matplotlib.pyplot as plt
from support.colortrans import pg_get_cmap
import threading

# -----------------------------------------------
from UI_interface import MiniStatusWindow, Ui_MainWindow
# -----------------------------------------------


# comments by ZHChen 2025-05-20

# 全局变量初始化
CLIport_name = ''
Dataport_name = ''
runtime_cfg = None
collector = None
processor = None
capture_started = False
DEFAULT_CFG_PATH = os.path.join('config', 'Radar.cfg')
last_fall_alert_result = {
    'source': 'ml',
    'detected': False,
    'strategy': 'ml_primary',
    'message': 'Standby',
    'metrics': {},
    'metadata': {},
}

tracked_target_state = tt.reset_tracker()
_TRACKER_GATE_M = 1.0
_TRACKER_MAX_MISS = 2
_TRACKER_STABLE_HITS = 2
_TRACKER_TIMEOUT_MS = 2000
_TRACKER_RELOCK_GRACE_MS = 1000
_TRACKER_RELOCK_DISTANCE_M = 0.6

# fall_predictor 实例（默认 NullFallPredictor）
_fall_predictor = fp.build_fall_predictor()
_last_ml_prediction = fp.FallPrediction(available=False)  # 上次 ML 结果
_last_logged_ml_token = None
_loaded_model_path = ""
_ml_load_message = 'No model loaded. Supports RACA checkpoints (*.pth), TorchScript (*.pt/*.jit/*.ts), and predictor plugins (*.py).'
_ml_contract = None
_ml_contract_validation = fp.ContractValidationResult(
    valid=False,
    status='disabled',
    message='ML disabled',
)
_ml_dropped_ml_frames = 0
_ml_recent_lagging = False
_ml_last_error = ""
_ml_inference_history = deque(maxlen=60)
_ml_alarm_state = mla.build_initial_alarm_state(
    enabled=True,
    model_loaded=False,
    contract_valid=False,
    contract_status='disabled',
    required_streak=2,
    threshold=0.5,
)
_replay_source = rc.NullReplaySource()
_loaded_replay_path = ""
_latest_aligned_ml_frame = None
_realtime_heatmaps_enabled = False
realtime_dashboard_widget = None
realtime_dashboard_layout = None
realtime_model_status_label = None


def _get_fall_status_style(level='idle'):
    palette = {
        'idle': ('#f5f5f5', '#9e9e9e', '#333333'),
        'monitor': ('#e8f4ff', '#4a90e2', '#0b3d91'),
        'capture': ('#fff4e5', '#f5a623', '#8a5a00'),
        'alert': ('#ffe9e9', '#d0021b', '#7f0d19'),
    }
    background, border, foreground = palette.get(level, palette['idle'])
    return (
        "border-width: 2px;"
        "border-style: solid;"
        f"border-color: {border};"
        f"background-color: {background};"
        f"color: {foreground};"
        "font-size: 22px;"
        "font-weight: bold;"
        "padding: 12px;"
    )


def _set_fall_status_preview(message, level='idle'):
    text = str(message).strip() or 'Standby'

    if 'fall_status_view' in globals() and fall_status_view is not None:
        fall_status_view.setText(text)
        fall_status_view.setAlignment(QtCore.Qt.AlignCenter)
        fall_status_view.setWordWrap(True)
        fall_status_view.setStyleSheet(_get_fall_status_style(level))

    if 'subWin' in globals() and subWin is not None and hasattr(subWin, 'set_status'):
        subWin.set_status(text, level)


def _set_runtime_aux_status(label_widget, message, level='idle'):
    if label_widget is None:
        return

    palette = {
        'idle': ('#f8f8f8', '#d0d0d0', '#555555'),
        'ready': ('#eef8ef', '#6aa84f', '#2e6b2e'),
        'warn': ('#fff8e8', '#f0ad4e', '#8a5a00'),
    }
    background, border, foreground = palette.get(level, palette['idle'])
    label_widget.setText(str(message).strip())
    label_widget.setWordWrap(True)
    label_widget.setStyleSheet(
        "border-width: 1px;"
        "border-style: solid;"
        f"border-color: {border};"
        f"background-color: {background};"
        f"color: {foreground};"
        "padding: 6px;"
    )


def _record_ml_inference_history(prediction):
    if prediction is None or not getattr(prediction, 'available', False):
        return

    metadata = dict(getattr(prediction, 'metadata', {}) or {})
    inference_id = metadata.get('inference_id')
    if inference_id is None:
        return
    try:
        inference_id = int(inference_id)
    except Exception:
        return
    frame_index = metadata.get('frame_index', inference_id)
    try:
        frame_index = int(frame_index)
    except Exception:
        frame_index = inference_id

    probability = float(getattr(prediction, 'probability', 0.0) or 0.0)
    label = str(getattr(prediction, 'label', '') or 'unknown')

    if _ml_inference_history and _ml_inference_history[-1].get('inference_id') == inference_id:
        _ml_inference_history[-1] = {
            'inference_id': inference_id,
            'frame_index': frame_index,
            'probability': probability,
            'label': label,
        }
        return

    _ml_inference_history.append(
        {
            'inference_id': inference_id,
            'frame_index': frame_index,
            'probability': probability,
            'label': label,
        }
    )



def _update_realtime_detection_dashboard():
    if 'realtime_prob_bar' in globals() and realtime_prob_bar is not None:
        probability = int(round(max(0.0, min(1.0, float(_ml_alarm_state.last_probability))) * 100))
        realtime_prob_bar.setValue(probability)
        realtime_prob_bar.setFormat(f"Fall probability {probability}%")

    if 'realtime_confirm_bar' in globals() and realtime_confirm_bar is not None:
        required = max(1, int(_ml_alarm_state.required_streak))
        streak = min(required, max(0, int(_ml_alarm_state.streak)))
        realtime_confirm_bar.setMaximum(required)
        realtime_confirm_bar.setValue(streak)
        realtime_confirm_bar.setFormat(f"Confirmations {streak}/{required}")

    if 'realtime_gate_label' in globals() and realtime_gate_label is not None:
        gate_ready = _has_stable_tracked_target()
        if gate_ready:
            gate_text = "ready"
        elif tracked_target_state.locked and tracked_target_state.position is not None:
            gate_text = f"warming ({tracked_target_state.hit_streak}/{_TRACKER_STABLE_HITS})"
        else:
            gate_text = "waiting for stable target"
        gate_level = 'ready' if gate_ready else 'warn'
        _set_runtime_aux_status(realtime_gate_label, f"Target gate: {gate_text}", level=gate_level)

    if 'realtime_summary_label' in globals() and realtime_summary_label is not None:
        lines = [
            f"ML state: {_ml_alarm_state.message}",
            f"Alarm state: {_ml_alarm_state.status}",
            f"Decision: threshold={_get_ml_threshold():.2f}, confirmations={_get_ml_required_streak()}",
            _get_tracker_status_text(),
        ]
        if _ml_alarm_state.last_inference_id is not None:
            latest_frame_index = (_last_ml_prediction.metadata or {}).get('frame_index')
            frame_segment = f", frame={latest_frame_index}" if latest_frame_index is not None else ""
            lines.append(
                f"Latest inference: id={_ml_alarm_state.last_inference_id}{frame_segment}, "
                f"label={_ml_alarm_state.last_label or 'unknown'}, "
                f"p={_ml_alarm_state.last_probability:.2f}"
            )
        else:
            lines.append("Latest inference: waiting for model output")

        if tracked_target_state.position is not None:
            tx, ty, tz, tr = tracked_target_state.position
            lines.append(f"Target: x={tx:.2f}m y={ty:.2f}m z={tz:.2f}m r={tr:.2f}m")
        else:
            lines.append("Target: no stable target")

        lines.append(f"Contract: {_ml_contract_validation.message}")
        summary_text = "\n".join(lines)
        if hasattr(realtime_summary_label, 'setPlainText'):
            realtime_summary_label.setPlainText(summary_text)
        else:
            realtime_summary_label.setText(summary_text)

    if (
        'realtime_prob_curve' in globals() and realtime_prob_curve is not None
        and 'realtime_prob_scatter' in globals() and realtime_prob_scatter is not None
    ):
        if _ml_inference_history:
            xs = np.array([item['frame_index'] for item in _ml_inference_history], dtype=np.float32)
            ys = np.array([item['probability'] for item in _ml_inference_history], dtype=np.float32)
            brushes = [
                pg.mkBrush('#d0021b' if item['label'].lower() == 'fall' else '#4a90e2')
                for item in _ml_inference_history
            ]
            realtime_prob_curve.setData(xs, ys)
            realtime_prob_scatter.setData(
                x=xs,
                y=ys,
                brush=brushes,
                size=10,
                pen=pg.mkPen('#ffffff', width=1),
            )
        else:
            realtime_prob_curve.setData([], [])
            realtime_prob_scatter.setData([], [])

def _has_loaded_model():
    return bool(_loaded_model_path)


def _is_ml_detection_enabled():
    if 'fall_detection_enabled' in globals() and fall_detection_enabled is not None:
        return bool(fall_detection_enabled.isChecked())
    return True


def _get_ml_threshold():
    if 'ui' in globals() and hasattr(ui, 'doubleSpinBox_fall_height_threshold'):
        try:
            return float(ui.doubleSpinBox_fall_height_threshold.value())
        except Exception:
            pass
    return float(getattr(_ml_alarm_state, 'threshold', 0.5))


def _get_ml_required_streak():
    if 'ui' in globals() and hasattr(ui, 'doubleSpinBox_fall_time_window'):
        try:
            return max(1, int(round(float(ui.doubleSpinBox_fall_time_window.value()))))
        except Exception:
            pass
    return int(getattr(_ml_alarm_state, 'required_streak', 2))


def _has_stable_tracked_target():
    return tt.is_tracker_stable(
        tracked_target_state,
        min_hits=_TRACKER_STABLE_HITS,
        max_miss=_TRACKER_MAX_MISS,
    )


def _get_tracker_status_text():
    return tt.format_tracker_status(
        tracked_target_state,
        max_miss=_TRACKER_MAX_MISS,
        stable_hits=_TRACKER_STABLE_HITS,
    )


def _default_fall_alert_result(message='Standby'):
    return {
        'source': 'ml',
        'detected': False,
        'strategy': 'ml_primary',
        'message': str(message),
        'metrics': {},
        'metadata': {},
    }


def _tracker_event_requires_ml_reset(tracker_event, tracker_state):
    if tracker_event == 'lost':
        return True
    if tracker_event == 'relocked' and not bool(getattr(tracker_state, 'stable_latched', False)):
        return True
    return False


def _refresh_monitor_status_preview():
    monitoring_active = 'monitor_button' in globals() and monitor_button is not None and monitor_button.isChecked()

    if not monitoring_active:
        _set_fall_status_preview("Standby\nClick 'Start Monitoring' to enter realtime detection", level='idle')
        return

    if _ml_alarm_state.active:
        _set_fall_status_preview(
            f"Fall Alert\n{last_fall_alert_result.get('message', _ml_alarm_state.message)}",
            level='alert',
        )
        return

    if _has_stable_tracked_target():
        status_lines = ["Tracking"] + _build_ml_status_lines()
        _set_fall_status_preview("\n".join(status_lines), level='monitor')
        return

    if tracked_target_state.locked and tracked_target_state.position is not None:
        status_lines = [f"Target Warmup ({tracked_target_state.hit_streak}/{_TRACKER_STABLE_HITS})"] + _build_ml_status_lines()
        _set_fall_status_preview("\n".join(status_lines), level='monitor')
        return

    status_lines = ["No Target"] + _build_ml_status_lines()
    _set_fall_status_preview("\n".join(status_lines), level='idle')


def _set_monitor_enabled(enabled):
    if 'monitor_button' not in globals() or monitor_button is None:
        return

    enabled = bool(enabled)
    monitor_button.blockSignals(True)
    monitor_button.setChecked(enabled)
    monitor_button.blockSignals(False)

    if 'ui' in globals() and ui is not None and hasattr(ui, 'toggle_monitor_button'):
        ui.toggle_monitor_button(force_off=not enabled)

    _refresh_monitor_status_preview()


def _sync_ml_alarm_state(prediction=None):
    global _ml_alarm_state, last_fall_alert_result

    _ml_alarm_state, alert_triggered = mla.update_alarm_state(
        _ml_alarm_state,
        prediction,
        enabled=_is_ml_detection_enabled(),
        model_loaded=_has_loaded_model(),
        contract_valid=_ml_contract_validation.valid,
        contract_status=_ml_contract_validation.status,
        tracker_ready=_has_stable_tracked_target(),
        threshold=_get_ml_threshold(),
        required_streak=_get_ml_required_streak(),
        lagging=_ml_recent_lagging,
        error_message=_ml_last_error,
    )

    if not _ml_alarm_state.active:
        last_fall_alert_result = _default_fall_alert_result(_ml_alarm_state.message)

    _update_realtime_detection_dashboard()
    return alert_triggered


def _ml_predicts_fall(prediction=None):
    prediction = prediction or _last_ml_prediction
    return mla.prediction_is_positive(
        prediction,
        threshold=_get_ml_threshold(),
    )


def _build_ml_status_lines():
    line = str(getattr(_ml_alarm_state, 'message', '') or '').strip()
    if not line:
        line = "ML disabled"
    return [line]


def _build_ml_log_token(prediction):
    metadata = getattr(prediction, "metadata", {}) or {}
    label = str(getattr(prediction, "label", "") or "").strip().lower()
    probability = round(float(getattr(prediction, "probability", 0.0) or 0.0), 2)
    warning = str(metadata.get("warning", "") or "").strip().lower()

    if _ml_last_error:
        return ("error", str(_ml_last_error))
    if not getattr(prediction, "available", False):
        return ("state", _ml_alarm_state.status, _ml_alarm_state.message, warning)
    if _ml_alarm_state.status in {"positive_pending", "alert"}:
        return ("alarm", _ml_alarm_state.status, label, probability, int(_ml_alarm_state.streak))
    if label == "fall":
        return ("positive", _ml_alarm_state.status, label, probability)
    return ("state", _ml_alarm_state.status, label, warning)


def _format_ml_log_message(prediction):
    metadata = getattr(prediction, "metadata", {}) or {}
    frame_index = metadata.get("frame_index")
    frame_segment = f" | frame={frame_index}" if frame_index is not None else ""

    if _ml_last_error:
        return f"[ML] error: {_ml_last_error}"
    if prediction is None or not getattr(prediction, "available", False):
        return f"[ML] {_ml_alarm_state.message}"
    return (
        f"[ML] {_ml_alarm_state.message}"
        f"{frame_segment} | label={prediction.label or 'unknown'}"
        f" | p={float(getattr(prediction, 'probability', 0.0) or 0.0):.2f}"
    )


def _drain_all_frames(queue_obj):
    items = []
    while True:
        try:
            items.append(queue_obj.get_nowait())
        except Empty:
            break
        except Exception:
            break
    return items


def _merge_ml_prediction_metadata(prediction):
    global _last_ml_prediction

    metadata = dict(prediction.metadata or {})
    metadata['contract_status'] = _ml_contract_validation.status
    metadata['dropped_ml_frames'] = _ml_dropped_ml_frames
    if _ml_recent_lagging:
        metadata.setdefault('warning', 'ML lagging')
    else:
        metadata.pop('warning', None)
    if _ml_last_error:
        metadata['error'] = _ml_last_error
    else:
        metadata.pop('error', None)
    prediction.metadata = metadata
    _last_ml_prediction = prediction
    return prediction


def _update_model_status_label():
    status_widgets = []
    if 'model_status_label' in globals() and model_status_label is not None:
        status_widgets.append(model_status_label)
    if 'realtime_model_status_label' in globals() and realtime_model_status_label is not None:
        status_widgets.append(realtime_model_status_label)
    if not status_widgets:
        return

    if not _has_loaded_model():
        for widget in status_widgets:
            _set_runtime_aux_status(widget, _ml_load_message, level='idle')
        return

    lines = [_ml_load_message]
    if _ml_contract is not None:
        lines.append(f"Contract: {fp.format_contract_summary(_ml_contract)}")
    if _ml_contract_validation is not None:
        lines.append(_ml_contract_validation.message)
        if _ml_contract_validation.mismatches:
            lines.append("Mismatch: " + "; ".join(_ml_contract_validation.mismatches))
    lines.append(
        f"Decision: threshold={_get_ml_threshold():.2f}, confirmations={_get_ml_required_streak()}"
    )
    lines.append(f"State: {_ml_alarm_state.message}")
    lines.append(_get_tracker_status_text())
    if _ml_alarm_state.last_inference_id is not None:
        latest_frame_index = (_last_ml_prediction.metadata or {}).get('frame_index')
        frame_segment = f", frame={latest_frame_index}" if latest_frame_index is not None else ""
        lines.append(
            f"Latest inference: id={_ml_alarm_state.last_inference_id}{frame_segment}, "
            f"label={_ml_alarm_state.last_label or 'unknown'}, "
            f"p={float(_ml_alarm_state.last_probability):.2f}"
        )
    else:
        metadata = _last_ml_prediction.metadata or {}
        buffered_frames = int(metadata.get('buffered_frames', 0))
        required_frames = int(metadata.get('required_frames', getattr(_ml_contract, 'clip_frames', 0) or 100))
        if buffered_frames > 0:
            lines.append(f"Warming: {buffered_frames}/{required_frames}")
    if _ml_dropped_ml_frames > 0:
        lines.append(f"Dropped ML frames: {_ml_dropped_ml_frames}")
    if _ml_last_error:
        lines.append(f"Last error: {_ml_last_error}")
    if tracked_target_state.position is not None:
        tx, ty, tz, tr = tracked_target_state.position
        lines.append(f"Tracked target: x={tx:.2f} y={ty:.2f} z={tz:.2f} r={tr:.2f}")

    level = 'ready'
    if not _ml_contract_validation.valid or _ml_last_error:
        level = 'warn'
    status_text = "\n".join(lines)
    for widget in status_widgets:
        _set_runtime_aux_status(widget, status_text, level=level)


def _refresh_ml_contract_state(load_message=None, contract=None, validation=None):
    global _ml_load_message, _ml_contract, _ml_contract_validation

    if load_message is not None:
        _ml_load_message = load_message

    if not _has_loaded_model():
        _ml_contract = None
        _ml_contract_validation = fp.ContractValidationResult(
            valid=False,
            status='disabled',
            message='ML disabled',
        )
        _sync_ml_alarm_state()
        _update_model_status_label()
        return

    if contract is None:
        try:
            contract = fp.load_model_runtime_contract(_loaded_model_path, runtime_cfg=runtime_cfg)
        except Exception as exc:
            contract = None
            validation = fp.ContractValidationResult(
                valid=False,
                status='error',
                message=f"ML contract error: {exc}",
                mismatches=[str(exc)],
            )

    _ml_contract = contract

    if validation is None:
        validation = fp.validate_runtime_contract(_ml_contract, runtime_cfg)

    _ml_contract_validation = validation
    _sync_ml_alarm_state()
    _update_model_status_label()


def _reset_ml_runtime_pipeline(reset_predictor=False):
    global _last_ml_prediction, _last_logged_ml_token, _ml_dropped_ml_frames
    global _ml_recent_lagging, _ml_last_error, _latest_aligned_ml_frame

    if 'MLFeatureData' in globals() and MLFeatureData is not None:
        _clear_queue(MLFeatureData)

    _ml_dropped_ml_frames = 0
    _ml_recent_lagging = False
    _ml_last_error = ""
    _last_logged_ml_token = None
    _latest_aligned_ml_frame = None
    _ml_inference_history.clear()
    _last_ml_prediction = fp.FallPrediction(available=False)
    _merge_ml_prediction_metadata(_last_ml_prediction)

    if reset_predictor and hasattr(_fall_predictor, 'reset'):
        _fall_predictor.reset()

    _sync_ml_alarm_state(_last_ml_prediction)
    _update_model_status_label()


def _build_current_pointcloud_snapshot():
    if len(PointCloudHistory) == 0:
        return None
    pointcloud_frames = [pc for _, pc in PointCloudHistory if pc is not None and len(pc) > 0]
    if not pointcloud_frames:
        return None
    return np.concatenate(pointcloud_frames, axis=0).astype(np.float32)


def _consume_ml_feature_frames():
    global _last_logged_ml_token, _ml_dropped_ml_frames, _ml_recent_lagging
    global _ml_last_error, last_fall_alert_result, _latest_aligned_ml_frame

    if 'MLFeatureData' not in globals() or MLFeatureData is None:
        return

    if not _has_loaded_model():
        _clear_queue(MLFeatureData)
        _merge_ml_prediction_metadata(fp.FallPrediction(available=False))
        _sync_ml_alarm_state(_last_ml_prediction)
        _update_model_status_label()
        return

    if not _ml_contract_validation.valid:
        _clear_queue(MLFeatureData)
        _merge_ml_prediction_metadata(
            fp.FallPrediction(
                available=False,
                metadata={'reason': _ml_contract_validation.message},
            )
        )
        _sync_ml_alarm_state(_last_ml_prediction)
        _update_model_status_label()
        return

    ml_frames = _drain_all_frames(MLFeatureData)
    if not ml_frames:
        _update_model_status_label()
        return

    pointcloud_snapshot = _build_current_pointcloud_snapshot()

    for ml_frame in ml_frames:
        _latest_aligned_ml_frame = ml_frame
        frame_metadata = dict(getattr(ml_frame, 'metadata', {}) or {})
        _ml_recent_lagging = bool(frame_metadata.get('dropped_on_enqueue')) or (
            str(frame_metadata.get('warning', '')).strip().lower() == 'ml lagging'
        )
        _ml_dropped_ml_frames = max(
            _ml_dropped_ml_frames,
            int(frame_metadata.get('dropped_ml_frames', 0) or 0),
        )
        try:
            feature_clip = _build_feature_clip(
                ml_frame.timestamp_end_ms,
                None,
                None,
                ml_frame.RDT,
                ml_frame.ART,
                ml_frame.ERT,
                pointcloud=pointcloud_snapshot,
            )
            prediction = _fall_predictor.predict(feature_clip)
            merged_metadata = dict(prediction.metadata or {})
            merged_metadata.update(frame_metadata)
            prediction.metadata = merged_metadata
            _ml_last_error = ""
            _merge_ml_prediction_metadata(prediction)
            _record_ml_inference_history(prediction)
            alert_triggered = _sync_ml_alarm_state(prediction)
            current_ml_log_token = _build_ml_log_token(prediction)
            if current_ml_log_token != _last_logged_ml_token:
                print(_format_ml_log_message(prediction))
                _last_logged_ml_token = current_ml_log_token
            if alert_triggered and 'monitor_button' in globals() and monitor_button.isChecked():
                last_fall_alert_result = mla.build_alert_result(_ml_alarm_state, prediction)
                _show_fall_alert(last_fall_alert_result, ml_frame.timestamp_end_ms)
        except Exception as ml_exc:
            _ml_last_error = str(ml_exc)
            _ml_recent_lagging = False
            print(f"[ML] inference error: {ml_exc}")
            _merge_ml_prediction_metadata(
                fp.FallPrediction(
                    available=False,
                    metadata={'reason': 'inference error'},
                )
            )
            _sync_ml_alarm_state(_last_ml_prediction)
            _last_logged_ml_token = None
            break

    _update_model_status_label()

def _clear_queue(queue_obj):
    while True:
        try:
            queue_obj.get_nowait()
        except Exception:
            break


def _drain_latest_frame(queue_obj, previous=None):
    latest = previous
    while True:
        try:
            latest = queue_obj.get_nowait()
        except Empty:
            break
        except Exception:
            break
    return latest


def _discover_cfg_candidates():
    config_dir = os.path.dirname(DEFAULT_CFG_PATH)
    if not os.path.isdir(config_dir):
        return []
    return sorted(
        os.path.join(config_dir, name)
        for name in os.listdir(config_dir)
        if name.lower().endswith('.cfg')
    )


def get_active_config_path():
    if 'radarparameters' in globals() and radarparameters is not None:
        current_text = radarparameters.currentText()
        if current_text and current_text != '--select--':
            return current_text
    if os.path.exists(DEFAULT_CFG_PATH):
        return DEFAULT_CFG_PATH
    candidates = _discover_cfg_candidates()
    return candidates[0] if candidates else None


def _populate_cfg_selector():
    if 'ui' not in globals() or ui is None:
        return
    ui.configpath()
    selected_path = get_active_config_path()
    if 'radarparameters' in globals() and radarparameters is not None and selected_path:
        index = radarparameters.findText(selected_path)
        if index >= 0:
            radarparameters.setCurrentIndex(index)


def _update_runtime_ui(config_params):
    range_resolution_m = float(config_params.get('range_resolution_m', config_params.get('rangeResolutionMeters', 0.0)))
    doppler_resolution_mps = float(config_params.get('doppler_resolution_mps', config_params.get('dopplerResolutionMps', 0.0)))
    max_range = float(config_params.get('maxRange', 0.0))
    max_velocity = float(config_params.get('maxVelocity', 0.0))

    if 'rangeResolutionlabel' in globals() and rangeResolutionlabel is not None:
        rangeResolutionlabel.setText(f"{range_resolution_m * 100:.2f}cm")
    if 'dopplerResolutionlabel' in globals() and dopplerResolutionlabel is not None:
        dopplerResolutionlabel.setText(f"{doppler_resolution_mps:.2f}m/s")
    if 'maxRangelabel' in globals() and maxRangelabel is not None:
        maxRangelabel.setText(f"{max_range:.2f}m")
    if 'maxVelocitylabel' in globals() and maxVelocitylabel is not None:
        maxVelocitylabel.setText(f"{max_velocity:.2f}m/s")

    if 'trajectory_plot_widget' in globals() and trajectory_plot_widget is not None:
        trajectory_plot_widget.setTitle(
            f"目标运动轨迹 (XY平面) | 距离分辨率: {range_resolution_m * 100:.1f}cm | 范围: ±{max_range:.1f}m"
        )
        trajectory_plot_widget.setXRange(-max_range, max_range)
        trajectory_plot_widget.setYRange(-max_range, max_range)

    if 'height_plot_widget' in globals() and height_plot_widget is not None:
        height_plot_widget.setTitle(
            f"高度检测 (ZY平面) | 距离分辨率: {range_resolution_m * 100:.1f}cm | 高度范围: ±1.0m"
        )
        height_plot_widget.setXRange(-max_range, max_range)


def apply_runtime_cfg(config_file=None):
    global runtime_cfg

    config_file = config_file or get_active_config_path()
    if not config_file or not os.path.exists(config_file):
        return None

    if capture_started and runtime_cfg is not None:
        active_file = runtime_cfg.get('config_file')
        if active_file and os.path.normpath(config_file) != os.path.normpath(active_file):
            _update_runtime_ui(runtime_cfg)
            return runtime_cfg

    config_params = readpoint.IWR6843AOP_TLV(connect=False)._initialize(config_file=config_file)
    runtime_cfg = config_params
    DSP.apply_runtime_config(config_params)
    rtp.apply_runtime_config(config_params)
    _update_runtime_ui(config_params)
    _refresh_ml_contract_state()
    return runtime_cfg


def _build_runtime_workers():
    global collector, processor, tracked_target_state

    config_params = runtime_cfg or apply_runtime_cfg()
    if config_params is None:
        raise ValueError("未找到可用的雷达配置文件")

    DSP.apply_runtime_config(config_params)
    rtp.apply_runtime_config(config_params)

    for queue_obj in (BinData, PointCloudData, MLFeatureData):
        _clear_queue(queue_obj)
    PointCloudHistory.clear()
    tracked_target_state = tt.reset_tracker()
    _reset_ml_runtime_pipeline(reset_predictor=True)

    collector = UdpListener('Listener', BinData, int(config_params['frame_length']))
    processor = DataProcessor(
        'Processor',
        config_params,
        BinData,
        pointcloud_queue=PointCloudData,
        ml_feature_queue=MLFeatureData,
    )
    return collector, processor


def _get_ml_preset_name():
    if 'fall_new_strategy_sensitivity' in globals() and fall_new_strategy_sensitivity is not None:
        return str(fall_new_strategy_sensitivity.currentText()).strip()
    return "中等"


def _show_fall_alert(result, current_time_ms):
    global last_fall_alert_time

    if current_time_ms - last_fall_alert_time <= fall_alert_cooldown:
        return

    if 'fall_alert_label' in globals() and fall_alert_label is not None:
        fall_alert_label.setText("你跌倒了！")
        fall_alert_label.show()
        QtCore.QTimer.singleShot(
            3000,
            lambda: fall_alert_label.hide() if 'fall_alert_label' in globals() and fall_alert_label is not None else None,
        )

    _set_fall_status_preview(
        f"FALL ALERT\n{result.get('message', 'Fall detected')}",
        level='alert',
    )

    try:
        import winsound
        winsound.Beep(1000, 500)
    except Exception as exc:
        print(f"音效播放失败: {exc}")

    last_fall_alert_time = current_time_ms
    if str(result.get('source', '')).lower() == 'ml':
        metrics = result.get('metrics', {})
        print(f"[ML ALERT] {result.get('message', 'ML confirmed fall')}")
        print(
            f"  probability={metrics.get('probability', 0.0):.2f}, "
            f"streak={metrics.get('streak', 0)}/{metrics.get('required_streak', 0)}, "
            f"threshold={metrics.get('threshold', 0.0):.2f}, "
            f"inference_id={metrics.get('inference_id', 'n/a')}"
        )
        return
    metrics = result.get('metrics', {})
    print(f"[跌倒检测] {result.get('message', '检测到跌倒！')}")
    print(f"  高度: {metrics.get('max_height', 0.0):.2f}m → {metrics.get('min_height', 0.0):.2f}m (下降 {metrics.get('height_drop', 0.0):.2f}m)")
    print(f"  速度: {metrics.get('drop_velocity', 0.0):.2f} m/s, 加速度: {metrics.get('acceleration', 0.0):.2f} m/s²")
    print(f"  水平位移: {metrics.get('y_change', 0.0):.2f}m, 低姿态持续时间: {metrics.get('low_height_duration', 0.0):.2f}s")
    print(
        f"  满足条件: {metrics.get('satisfied_conditions', 0)}/{metrics.get('total_conditions', 0)} "
        f"(需要至少{metrics.get('min_conditions', 0)}个)"
    )


def _serialize_tracker_state():
    return tt.tracker_state_to_dict(tracked_target_state)


def _build_feature_clip(current_time_ms, rt_feature, dt_feature, rdt_feature, art_feature, ert_feature, pointcloud=None):
    return fp.FallFeatureClip(
        timestamp_start_ms=current_time_ms,
        timestamp_end_ms=current_time_ms,
        RT=rt_feature,
        DT=dt_feature,
        RDT=rdt_feature,
        ART=art_feature,
        ERT=ert_feature,
        pointcloud=pointcloud,
        height_history=list(height_history),
        tracked_target_state=_serialize_tracker_state(),
        runtime_cfg=dict(runtime_cfg) if runtime_cfg else {},
    )

def update_figure():
    global img_rdi, img_rai, img_rti, img_rei, img_dti
    global last_fall_alert_result
    global pointcloud_scatter, pointcloud_info_label, PointCloudData
    global pointcloud_max_points, pointcloud_refresh_rate, pointcloud_threshold
    global pointcloud_show_grid, pointcloud_show_axes
    global pointcloud_grid, pointcloud_grid_xy, pointcloud_grid_xz, pointcloud_grid_yz
    global pointcloud_x_axis, pointcloud_y_axis, pointcloud_z_axis
    global pointcloud_x_marker, pointcloud_y_marker, pointcloud_z_marker
    global pointcloud_last_update_time, pointcloud_refresh_interval
    global pointcloud_log_text, pointcloud_coordinate_info_label
    global PointCloudHistory
    global pointcloud_enable_clustering, pointcloud_cluster_eps, pointcloud_cluster_min_samples
    global trajectory_history, trajectory_plot, trajectory_info_label, trajectory_line_ref, trajectory_plot_widget
    global trajectory_history_length, trajectory_point_size, trajectory_show_axes, trajectory_show_grid
    global trajectory_view, trajectory_line_items, trajectory_scatter_items
    global target_count, trajectory_line_items, trajectory_scatter_items
    global height_history, height_plot, height_info_label, height_line_ref
    global height_history_length, height_point_size, height_show_axes, height_show_grid
    global fall_alert_label, last_fall_alert_time, fall_alert_cooldown
    global fall_detection_enabled, fall_new_strategy_sensitivity
    global tracked_target_state, _fall_predictor, _last_ml_prediction
    global _last_logged_ml_token, _ml_dropped_ml_frames, _ml_last_error
    global _TRACKER_GATE_M, _TRACKER_MAX_MISS, _TRACKER_TIMEOUT_MS

    # 时间-距离图绘制，向img_rti容器中添加需要绘制的数据，其中：
    # （1）RTIData.get()返回图像矩阵，图像由processor线程采集
    # （2）.sum(2)是对第三维求和（比如多个通道或Tx/Rx叠加）
    # （3）[0:1024:16,:]是抽稀处理，仅取部分数据减少渲染负担，16为稀疏化步长，已改成1
    # （4）img_rti.setImage(...)直接设置图像内容，立即生效
    # （5）使用QTimer.singleShot实现递归刷新机制，约每1ms调用一次，构成实时绘图。
    # （6）levels=[0, 1e4]是colorbar着色范围
    # 更新点云显示（带刷新速率控制）
    current_time = time.time() * 1000  # 转换为毫秒
    current_tracked_target = None
    current_frame_pointcloud = None
    pointcloud_refresh_interval = pointcloud_refresh_rate.value()
    
    # 控制网格和坐标系的显示
    if pointcloud_show_grid.isChecked():
        # 显示所有三个平面的网格
        try:
            if 'pointcloud_grid_xy' in globals() and pointcloud_grid_xy is not None:
                pointcloud_grid_xy.show()
            if 'pointcloud_grid_xz' in globals() and pointcloud_grid_xz is not None:
                pointcloud_grid_xz.show()
            if 'pointcloud_grid_yz' in globals() and pointcloud_grid_yz is not None:
                pointcloud_grid_yz.show()
            # 兼容旧代码
            if 'pointcloud_grid' in globals() and pointcloud_grid is not None:
                pointcloud_grid.show()
        except:
            pass
    else:
        # 隐藏所有网格
        try:
            if 'pointcloud_grid_xy' in globals() and pointcloud_grid_xy is not None:
                pointcloud_grid_xy.hide()
            if 'pointcloud_grid_xz' in globals() and pointcloud_grid_xz is not None:
                pointcloud_grid_xz.hide()
            if 'pointcloud_grid_yz' in globals() and pointcloud_grid_yz is not None:
                pointcloud_grid_yz.hide()
            # 兼容旧代码
            if 'pointcloud_grid' in globals() and pointcloud_grid is not None:
                pointcloud_grid.hide()
        except:
            pass
    
    if pointcloud_show_axes.isChecked():
        pointcloud_x_axis.show()
        pointcloud_y_axis.show()
        pointcloud_z_axis.show()
        pointcloud_x_marker.show()
        pointcloud_y_marker.show()
        pointcloud_z_marker.show()
    else:
        pointcloud_x_axis.hide()
        pointcloud_y_axis.hide()
        pointcloud_z_axis.hide()
        pointcloud_x_marker.hide()
        pointcloud_y_marker.hide()
        pointcloud_z_marker.hide()
    
    # 更新阈值参数到全局变量（供DSP使用）
    gl.set_value('pointcloud_threshold', pointcloud_threshold.value())
    
    # 根据刷新速率决定是否更新点云
    if current_time - pointcloud_last_update_time >= pointcloud_refresh_interval:
        pointcloud_last_update_time = current_time
        
        # 清空队列中所有旧数据，只保留最新的
        queue_size_before = PointCloudData.qsize()
        while PointCloudData.qsize() > 1:
            try:
                PointCloudData.get_nowait()
            except:
                break
        
        # 调试信息：每100次更新打印一次
        if hasattr(update_figure, '_debug_counter'):
            update_figure._debug_counter += 1
        else:
            update_figure._debug_counter = 0
        
        if getattr(rtp, 'POINTCLOUD_DEBUG_LOGS', False) and update_figure._debug_counter % 100 == 0:
            print(f"[PointCloud] queue size: {queue_size_before} -> {PointCloudData.qsize()}")
        
        # 处理新点云数据并加入历史缓冲区
        if not PointCloudData.empty():
            try:
                new_pointcloud = PointCloudData.get()
                if new_pointcloud is not None and len(new_pointcloud) > 0:
                    current_frame_pointcloud = np.asarray(new_pointcloud)
                    # 将新点云加入历史缓冲区（带时间戳）
                    PointCloudHistory.append((current_time, new_pointcloud))
                    
                    # 限制历史缓冲区大小（最多保留最近20帧）
                    if len(PointCloudHistory) > 20:
                        PointCloudHistory.pop(0)
            except Exception as e:
                print(f"处理新点云数据错误: {e}")
        
        # 清理过期的点云（基于能量的时间衰减）
        # 近距离点（高能量，红色）：保留时间长（500ms）
        # 远距离点（低能量，蓝色）：保留时间短（100ms）
        valid_history = []
        for timestamp, pointcloud in PointCloudHistory:
            age_ms = current_time - timestamp
            
            # 根据点的距离计算保留时间
            # 距离越近（能量越高），保留时间越长
            if len(pointcloud) > 0:
                ranges = pointcloud[:, 0]
                # 计算该帧点云的平均距离
                avg_range = np.mean(ranges)
                
                # 根据平均距离设置保留时间
                # 近距离（<1m）：保留1000ms
                # 中距离（1-3m）：保留300ms
                # 远距离（>3m）：保留100ms
                if avg_range < 1.0:
                    max_age_ms = 1000  # 高能量点保留1000ms
                elif avg_range < 3.0:
                    max_age_ms = 300  # 中能量点保留300ms
                else:
                    max_age_ms = 100  # 低能量点保留100ms
                
                # 只保留未过期的点云
                if age_ms < max_age_ms:
                    valid_history.append((timestamp, pointcloud))
        
        PointCloudHistory = valid_history
        
        # 合并所有有效的历史点云（带时间信息）
        if len(PointCloudHistory) > 0:
            try:
                # 合并所有历史点云，同时记录每个点的时间信息
                all_points_with_time = []
                for timestamp, pointcloud in PointCloudHistory:
                    if pointcloud is not None and len(pointcloud) > 0:
                        # 为每个点添加时间戳信息
                        num_points = len(pointcloud)
                        timestamps = np.full((num_points, 1), timestamp)
                        # 扩展点云格式: [range, x, y, z, timestamp]
                        points_with_time = np.hstack([pointcloud, timestamps])
                        all_points_with_time.append(points_with_time)
                
                if len(all_points_with_time) > 0:
                    # 合并所有点云（格式: [range, x, y, z, timestamp]）
                    merged_pointcloud = np.vstack(all_points_with_time)
                    
                    # 提取数据
                    ranges = merged_pointcloud[:, 0]  # 距离
                    x = merged_pointcloud[:, 1]       # X坐标
                    y = merged_pointcloud[:, 2]       # Y坐标
                    z = merged_pointcloud[:, 3]       # Z坐标
                    timestamps = merged_pointcloud[:, 4]  # 时间戳
                    
                    # 如果启用了聚类功能，进行聚类处理
                    # 检查变量是否已初始化（可能在application()函数初始化之前调用）
                    enable_clustering = False
                    cluster_eps_val = 0.3
                    cluster_min_samples_val = 3
                    is_clustered = False  # 初始化聚类状态标记
                    
                    try:
                        if 'pointcloud_enable_clustering' in globals() and pointcloud_enable_clustering is not None:
                            enable_clustering = pointcloud_enable_clustering.isChecked()
                            if 'pointcloud_cluster_eps' in globals() and pointcloud_cluster_eps is not None:
                                cluster_eps_val = pointcloud_cluster_eps.value()
                            if 'pointcloud_cluster_min_samples' in globals() and pointcloud_cluster_min_samples is not None:
                                cluster_min_samples_val = pointcloud_cluster_min_samples.value()
                    except (NameError, AttributeError):
                        # 变量未初始化，使用默认值（不启用聚类）
                        enable_clustering = False
                    if enable_clustering:
                        # 准备点云数据（格式: [range, x, y, z]）
                        pointcloud_data = merged_pointcloud[:, :4]
                        
                        # 执行聚类（使用之前获取的参数）
                        clustered_pointcloud, cluster_labels, cluster_info = pointcloud_clustering.cluster_pointcloud_simple(
                            pointcloud_data, 
                            eps=cluster_eps_val, 
                            min_samples=cluster_min_samples_val
                        )
                        
                        # 更新点云数据
                        if len(clustered_pointcloud) > 0:
                            ranges = clustered_pointcloud[:, 0]
                            x = clustered_pointcloud[:, 1]
                            y = clustered_pointcloud[:, 2]
                            z = clustered_pointcloud[:, 3]
                            # 聚类后的点没有时间戳，使用当前时间
                            timestamps = np.full(len(clustered_pointcloud), current_time)
                            is_clustered = True  # 标记为已聚类
                            
                            # 输出聚类信息（每100次更新打印一次）
                        if getattr(rtp, 'POINTCLOUD_DEBUG_LOGS', False) and update_figure._debug_counter % 100 == 0:
                                print(f"[点云聚类] 聚类前: {cluster_info['num_points_before']} 点 | "
                                      f"聚类后: {cluster_info['num_points_after']} 点 | "
                                      f"聚类数: {cluster_info['num_clusters']} | "
                                      f"噪声点: {cluster_info['num_noise']}")
                        else:
                            # 聚类后没有有效点，清空显示
                            ranges = np.array([])
                            x = np.array([])
                            y = np.array([])
                            z = np.array([])
                            timestamps = np.array([])
                            is_clustered = False
                    
                    # 限制点云数量（优先保留近距离的点）
                    max_points = pointcloud_max_points.value()
                    if len(x) > max_points:
                        # 按距离排序，优先保留近距离点（高能量点）
                        sorted_indices = np.argsort(ranges)
                        selected_indices = sorted_indices[:max_points]
                        ranges = ranges[selected_indices]
                        x = x[selected_indices]
                        y = y[selected_indices]
                        z = z[selected_indices]
                        timestamps = timestamps[selected_indices]
                    
                    # 改进的点云可视化：根据能量/幅值区分噪声点和目标点
                    # 噪声点：低能量（远距离），半透明浅蓝色，小尺寸
                    # 目标点：高能量（近距离/已聚类），橙色，大尺寸
                    if len(ranges) > 0:
                        # 计算能量阈值（使用距离作为能量代理）
                        # 近距离 = 高能量，远距离 = 低能量
                        energy_threshold = 2.0  # 2米作为能量阈值
                        
                        # 创建颜色数组和大小数组
                        color_array = np.zeros((len(ranges), 4))
                        size_array = np.zeros(len(ranges))
                        
                        for i in range(len(ranges)):
                            range_val = ranges[i]
                            age_ms = current_time - timestamps[i]
                            
                            # 判断是否为高能量点（目标点）
                            # 条件：近距离 OR 已聚类
                            is_target = (range_val < energy_threshold) or is_clustered
                            
                            if is_target:
                                # 目标点：橙色，不透明，大尺寸
                                # 橙色 RGB: (255, 165, 0) -> 归一化: (1.0, 0.65, 0.0)
                                color_array[i, 0] = 1.0   # R: 红色分量
                                color_array[i, 1] = 0  # G: 绿色分量
                                color_array[i, 2] = 0.0   # B: 蓝色分量
                                color_array[i, 3] = 0.7   # A: 完全不透明
                                size_array[i] = 9  # 大尺寸
                            else:
                                # 噪声点：浅蓝色，半透明，小尺寸
                                # 浅蓝色 RGB: (173, 216, 230) -> 归一化: (0.68, 0.85, 0.90)
                                color_array[i, 0] = 0.68  # R: 红色分量
                                color_array[i, 1] = 0.85  # G: 绿色分量
                                color_array[i, 2] = 0.90  # B: 蓝色分量
                                color_array[i, 3] = 0.3   # A: 半透明（30%透明度）
                                size_array[i] = 5  # 小尺寸
                            
                            # 根据时间衰减调整透明度（噪声点衰减更快）
                            if age_ms > 0:
                                if is_target:
                                    # 目标点：保留时间长，衰减慢
                                    max_age_ms = 1000 if range_val < 1.0 else 500
                                    if age_ms >= max_age_ms:
                                        color_array[i, 3] = 0.0
                                    else:
                                        decay_factor = 1.0 - (age_ms / max_age_ms)
                                        color_array[i, 3] *= max(0.7, decay_factor)
                                else:
                                    # 噪声点：保留时间短，快速衰减
                                    max_age_ms = 200
                                    if age_ms >= max_age_ms:
                                        color_array[i, 3] = 0.0
                                    else:
                                        decay_factor = 1.0 - (age_ms / max_age_ms)
                                        color_array[i, 3] *= decay_factor * 0.3
                    else:
                        # 如果没有点，创建空数组
                        color_array = np.ones((0, 4), dtype=np.float32) * 0.5  # 默认灰色
                        size_array = np.array([], dtype=np.float32)  # 空尺寸数组
                    
                    # 更新散点图
                    if len(x) > 0:
                        # 确保数据类型为 float32，pyqtgraph OpenGL 需要
                        pos = np.column_stack([x, y, z]).astype(np.float32)
                        # 确保 color_array 和 size_array 形状与 pos 匹配，并转换为 float32
                        if color_array.shape[0] != pos.shape[0]:
                            # 如果形状不匹配，创建默认颜色数组和尺寸数组
                            color_array = np.ones((pos.shape[0], 4), dtype=np.float32) * 0.5
                            size_array = np.full(pos.shape[0], 8, dtype=np.float32)
                        else:
                            color_array = color_array.astype(np.float32)
                            if len(size_array) != len(pos):
                                size_array = np.full(pos.shape[0], 8, dtype=np.float32)
                            else:
                                size_array = size_array.astype(np.float32)
                        
                        try:
                            # 使用动态大小数组（如果支持）或固定大小
                            # 注意：GLScatterPlotItem的size参数可以是标量或数组
                            if len(size_array) == len(pos) and len(size_array) > 0:
                                # 使用动态大小数组
                                pointcloud_scatter.setData(pos=pos, color=color_array, size=size_array)
                            else:
                                # 回退到固定大小
                                pointcloud_scatter.setData(pos=pos, color=color_array, size=8)
                        except Exception as e:
                            # 如果绘制失败，尝试清空并重新设置
                            print(f"点云绘制错误: {e}")
                            pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                                      color=np.empty((0, 4), dtype=np.float32), 
                                                      size=8)
                        
                        # 计算点云中心位置和方位信息（使用合并后的点云）
                        center_x = x.mean()
                        center_y = y.mean()
                        center_z = z.mean()
                        center_range = np.sqrt(center_x**2 + center_y**2 + center_z**2)
                    else:
                        # 如果没有点，清空散点图
                        try:
                            pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                                      color=np.empty((0, 4), dtype=np.float32), 
                                                      size=8)
                        except Exception as e:
                            print(f"清空点云错误: {e}")
                        # 设置默认值
                        center_x = center_y = center_z = center_range = 0.0
                    
                    # 计算方位角（azimuth，从X轴正方向逆时针，范围-180到180度）
                    azimuth_rad = np.arctan2(center_y, center_x)
                    azimuth_deg = np.rad2deg(azimuth_rad)
                    
                    # 计算俯仰角（elevation，从XY平面向上，范围-90到90度）
                    elevation_rad = np.arcsin(center_z / center_range) if center_range > 0.01 else 0
                    elevation_deg = np.rad2deg(elevation_rad)
                    
                    # 确定方位描述
                    def get_direction_description(azimuth_deg, elevation_deg):
                        """根据方位角和俯仰角返回方位描述"""
                        directions = []
                        
                        # 水平方位
                        if abs(azimuth_deg) < 22.5:
                            directions.append("前方")
                        elif abs(azimuth_deg - 180) < 22.5 or abs(azimuth_deg + 180) < 22.5:
                            directions.append("后方")
                        elif 67.5 < azimuth_deg < 112.5:
                            directions.append("左侧")
                        elif -112.5 < azimuth_deg < -67.5:
                            directions.append("右侧")
                        elif 22.5 <= azimuth_deg < 67.5:
                            directions.append("左前方")
                        elif 112.5 <= azimuth_deg < 157.5:
                            directions.append("左后方")
                        elif -157.5 <= azimuth_deg < -112.5:
                            directions.append("右后方")
                        elif -67.5 <= azimuth_deg < -22.5:
                            directions.append("右前方")
                        
                        # 垂直方位
                        if elevation_deg > 30:
                            directions.append("上方")
                        elif elevation_deg < -30:
                            directions.append("下方")
                        elif abs(elevation_deg) > 10:
                            if elevation_deg > 0:
                                directions.append("略上方")
                            else:
                                directions.append("略下方")
                        
                        return " ".join(directions) if directions else "中心"
                    
                    direction_desc = get_direction_description(azimuth_deg, elevation_deg)
                    
                    # 更新信息标签，显示四元组信息
                    if len(x) > 0:
                        range_info = f"距离范围: [{ranges.min():.2f}, {ranges.max():.2f}]m"
                        x_info = f"X范围: [{x.min():.2f}, {x.max():.2f}]m"
                        y_info = f"Y范围: [{y.min():.2f}, {y.max():.2f}]m"
                        z_info = f"Z范围: [{z.min():.2f}, {z.max():.2f}]m"
                        # 检查聚类状态（安全访问）
                        try:
                            clustering_status = " [已聚类]" if ('pointcloud_enable_clustering' in globals() and pointcloud_enable_clustering is not None and pointcloud_enable_clustering.isChecked()) else ""
                        except:
                            clustering_status = ""
                        pointcloud_info_label.setText(f"点云数量: {len(merged_pointcloud)} (显示: {len(x)}){clustering_status} | {range_info} | {x_info} | {y_info} | {z_info}")
                    else:
                        # 检查聚类状态（安全访问）
                        try:
                            clustering_status = " [已聚类]" if ('pointcloud_enable_clustering' in globals() and pointcloud_enable_clustering is not None and pointcloud_enable_clustering.isChecked()) else ""
                        except:
                            clustering_status = ""
                        pointcloud_info_label.setText(f"点云数量: {len(merged_pointcloud)} (显示: 0){clustering_status} | 无有效点云")
                    
                    # 更新坐标信息显示（已隐藏）
                    # coord_info = f"【IWR6843雷达坐标系】\n"
                    # coord_info += f"X轴(红色箭头) = 前方（雷达正对方向）\n"
                    # coord_info += f"Y轴(绿色箭头) = 左侧（方位向，8个阵元）\n"
                    # coord_info += f"Z轴(蓝色箭头) = 上方（俯仰向，2个阵元）\n"
                    # coord_info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    # coord_info += f"点云中心位置: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f}) m\n"
                    # coord_info += f"距离雷达: {center_range:.2f} m\n"
                    # coord_info += f"方位角: {azimuth_deg:+.1f}° (0°=前方, +90°=左侧, -90°=右侧, ±180°=后方)\n"
                    # coord_info += f"俯仰角: {elevation_deg:+.1f}° (0°=水平, +90°=正上方, -90°=正下方)\n"
                    # coord_info += f"方位描述: {direction_desc}"
                    # if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                    #     pointcloud_coordinate_info_label.setText(coord_info)
                    
                    # 输出详细日志信息
                    log_msg = f"点云采集 | 总数: {len(merged_pointcloud)} | 显示: {len(x)} | "
                    log_msg += f"距离: [{ranges.min():.2f}, {ranges.max():.2f}]m | "
                    log_msg += f"X: [{x.min():.2f}, {x.max():.2f}]m | "
                    log_msg += f"Y: [{y.min():.2f}, {y.max():.2f}]m | "
                    log_msg += f"Z: [{z.min():.2f}, {z.max():.2f}]m"
                    if getattr(rtp, 'POINTCLOUD_DEBUG_LOGS', False):
                        print_pointcloud_log(log_msg, fontcolor='blue')
                    
                    # -------------------------------------------------------
                    # 单目标稳定跟踪：从当前帧聚类候选中关联主目标
                    # -------------------------------------------------------
                    targets = []
                    
                    # 更新轨迹显示（找出能量最强的那一簇点云，加权计算中心）
                    try:
                        num_targets = target_count.value()
                        cluster_centers = []
                        if current_frame_pointcloud is not None and len(current_frame_pointcloud) > 0:
                            current_frame_points = np.asarray(current_frame_pointcloud, dtype=np.float32)

                            from sklearn.cluster import DBSCAN
                            spatial_coords = current_frame_points[:, 1:4]
                            cluster_eps_val = max(0.05, float(pointcloud_cluster_eps.value()))
                            cluster_min_samples_val = max(4, int(pointcloud_cluster_min_samples.value()))
                            clustering = DBSCAN(
                                eps=cluster_eps_val,
                                min_samples=cluster_min_samples_val,
                            )
                            cluster_labels = clustering.fit_predict(spatial_coords)

                            unique_labels = np.unique(cluster_labels)
                            cluster_centers = []
                            for label in unique_labels:
                                if label == -1:
                                    continue
                                cluster_mask = cluster_labels == label
                                cluster_points = current_frame_points[cluster_mask]
                                if len(cluster_points) < cluster_min_samples_val:
                                    continue
                                cluster_ranges = cluster_points[:, 0]
                                if float(np.min(cluster_ranges)) <= 0.25:
                                    continue
                                cluster_energies = 1.0 / (cluster_ranges + 0.01)
                                energy_sum = float(np.sum(cluster_energies))
                                if energy_sum <= 0.0:
                                    continue
                                weights = cluster_energies / cluster_energies.sum()
                                wx = np.average(cluster_points[:, 1], weights=weights)
                                wy = np.average(cluster_points[:, 2], weights=weights)
                                wz = np.average(cluster_points[:, 3], weights=weights)
                                wr = np.sqrt(wx**2 + wy**2 + wz**2)
                                cluster_centers.append((energy_sum, wx, wy, wz, wr, len(cluster_points)))
                            cluster_centers.sort(key=lambda c: c[0], reverse=True)

                            tracker_candidates = [
                                (cx, cy, cz, cr)
                                for _, cx, cy, cz, cr, _ in cluster_centers
                                if cr > 0.25
                            ]
                            tracked_target_state, tracker_event = tt.update_tracker(
                                tracked_target_state,
                                tracker_candidates,
                                current_time,
                                gate_m=_TRACKER_GATE_M,
                                max_miss=_TRACKER_MAX_MISS,
                                timeout_ms=_TRACKER_TIMEOUT_MS,
                                stable_hits=_TRACKER_STABLE_HITS,
                                relock_grace_ms=_TRACKER_RELOCK_GRACE_MS,
                                relock_distance_m=_TRACKER_RELOCK_DISTANCE_M,
                            )

                            if _tracker_event_requires_ml_reset(tracker_event, tracked_target_state):
                                height_history = []
                                _reset_ml_runtime_pipeline(reset_predictor=True)
                                last_fall_alert_result = _default_fall_alert_result('ML ready')

                            if tracker_event in ('locked', 'tracked', 'predicted', 'relocked') and tracked_target_state.position is not None:
                                current_tracked_target = tracked_target_state.position
                                targets = [current_tracked_target]

                            # 多目标轨迹显示仍保留（供轨迹页使用），从 cluster_centers 取前 num_targets 个
                            display_targets = []
                            for _, tx, ty, tz, tr, _ in cluster_centers[:num_targets]:
                                if tr > 0.01:
                                    display_targets.append((tx, ty, tz, tr))
                            if len(display_targets) < num_targets and tracked_target_state.position is not None:
                                display_targets.append(tracked_target_state.position)
                            elif len(display_targets) < num_targets and len(x) > 0:
                                display_targets.append((center_x, center_y, center_z, center_range))
                            display_targets = display_targets[:num_targets]
                            
                            # 添加新的轨迹点（使用 display_targets 供多目标轨迹显示）
                            # 轨迹历史存储格式: [(timestamp, target_id, x, y, range), ...]
                            for target_id, (target_x, target_y, target_z, target_range) in enumerate(display_targets):
                                trajectory_history.append((current_time, target_id, target_x, target_y, target_range))
                            
                            # 限制轨迹历史长度
                            max_history = trajectory_history_length.value()
                            if len(trajectory_history) > max_history:
                                trajectory_history = trajectory_history[-max_history:]
                            
                            # 提取轨迹数据（支持多目标）
                            if len(trajectory_history) > 0:
                                # 按目标ID分组轨迹
                                num_targets = target_count.value()
                                target_colors = [
                                    (255, 0, 0, 200),      # 目标0: 红色
                                    (0, 255, 0, 200),      # 目标1: 绿色
                                    (0, 0, 255, 200),      # 目标2: 蓝色
                                    (255, 255, 0, 200),    # 目标3: 黄色
                                    (255, 0, 255, 200),    # 目标4: 洋红
                                    (0, 255, 255, 200),    # 目标5: 青色
                                    (255, 128, 0, 200),    # 目标6: 橙色
                                    (128, 0, 255, 200),    # 目标7: 紫色
                                    (255, 192, 203, 200),   # 目标8: 粉色
                                    (128, 128, 128, 200),  # 目标9: 灰色
                                ]
                                
                                # 收集所有轨迹点
                                all_traj_x = []
                                all_traj_y = []
                                all_traj_ranges = []
                                
                                for target_id in range(num_targets):
                                    # 获取该目标的所有轨迹点
                                    target_points = [t for t in trajectory_history if len(t) > 1 and t[1] == target_id]
                                    
                                    if len(target_points) > 0:
                                        traj_x = [t[2] for t in target_points]  # X坐标（原始值，X轴正向朝右）
                                        traj_y = [-t[3] for t in target_points]  # Y坐标（翻转，使Y轴正向朝下，第一象限在右下角）
                                        traj_ranges = [t[4] for t in target_points]  # 距离
                                        
                                        all_traj_x.extend(traj_x)
                                        all_traj_y.extend(traj_y)
                                        all_traj_ranges.extend(traj_ranges)
                                
                                if len(all_traj_x) > 0:
                                    # 更新轨迹散点图（2D显示，使用渐变色：根据距离从红色到蓝色）
                                    # 根据距离计算渐变色（近处红色，远处蓝色）
                                    all_traj_ranges_array = np.array(all_traj_ranges)
                                    if len(all_traj_ranges_array) > 0:
                                        # 归一化距离到0-1范围
                                        if all_traj_ranges_array.max() > all_traj_ranges_array.min():
                                            normalized_ranges = (all_traj_ranges_array - all_traj_ranges_array.min()) / (all_traj_ranges_array.max() - all_traj_ranges_array.min())
                                        else:
                                            normalized_ranges = np.zeros_like(all_traj_ranges_array)
                                        
                                        # 生成渐变色：红色(近) -> 蓝色(远)
                                        # R: 从255到0，G: 从0到0，B: 从0到255
                                        color_list = []
                                        for norm_range in normalized_ranges:
                                            r = int(255 * (1 - norm_range))
                                            g = 0
                                            b = int(255 * norm_range)
                                            color_list.append((r, g, b, 200))  # 添加透明度
                                    else:
                                        # 如果没有距离数据，使用默认红色
                                        color_list = [(255, 0, 0, 200)] * len(all_traj_x)
                                    
                                    # 更新2D散点图（使用x和y坐标，颜色列表）
                                    trajectory_plot.setData(x=all_traj_x, y=all_traj_y, brush=color_list, size=trajectory_point_size.value())
                                    
                                    # 更新轨迹连线（显示所有点的连线）
                                    if len(all_traj_x) > 1:
                                        trajectory_line_ref.setData(x=all_traj_x, y=all_traj_y)
                                    else:
                                        trajectory_line_ref.setData([], [])
                                    
                                    # 清除旧的连线（不再需要，因为使用统一的连线）
                                    for line_item in trajectory_line_items:
                                        trajectory_plot_widget.removeItem(line_item)
                                    for scatter_item in trajectory_scatter_items:
                                        trajectory_plot_widget.removeItem(scatter_item)
                                    trajectory_line_items = []
                                    trajectory_scatter_items = []
                                
                                # 更新轨迹信息
                                if len(trajectory_history) > 0:
                                    # 获取最新一帧的所有目标位置
                                    latest_timestamp = max([t[0] for t in trajectory_history])
                                    latest_targets = [t for t in trajectory_history if t[0] == latest_timestamp]
                                    
                                    traj_info = f"目标数: {num_targets} | 轨迹点数: {len(trajectory_history)} | "
                                    if len(latest_targets) > 0:
                                        traj_info += "最新位置: "
                                        for i, t in enumerate(latest_targets[:num_targets]):
                                            target_id = t[1] if len(t) > 1 else i
                                            traj_x_original = t[2] if len(t) > 2 else 0  # X坐标（原始值）
                                            traj_y_original = t[3] if len(t) > 3 else 0  # Y坐标（原始值）
                                            traj_range = t[4] if len(t) > 4 else 0
                                            traj_info += f"目标{target_id}: ({traj_x_original:.2f}, {traj_y_original:.2f}) m "
                                        trajectory_info_label.setText(traj_info)
                                else:
                                    trajectory_info_label.setText("轨迹信息: 等待数据...")
                            else:
                                trajectory_plot.setData([], [])
                                trajectory_line_ref.setData([], [])
                                # 清除所有连线和散点图
                                if 'trajectory_line_items' in globals():
                                    for line_item in trajectory_line_items:
                                        if 'trajectory_plot_widget' in globals():
                                            trajectory_plot_widget.removeItem(line_item)
                                    trajectory_line_items = []
                                if 'trajectory_scatter_items' in globals():
                                    for scatter_item in trajectory_scatter_items:
                                        if 'trajectory_plot_widget' in globals():
                                            trajectory_plot_widget.removeItem(scatter_item)
                                    trajectory_scatter_items = []
                                trajectory_info_label.setText("轨迹信息: 等待数据...")
                            
                    except Exception as e:
                        # 轨迹更新失败不影响主流程
                        if 'trajectory_history' in globals():
                            pass
                        # 如果轨迹更新失败，targets可能为空，使用点云中心作为备选
                        if len(targets) == 0 and len(x) > 0:
                            # 使用点云中心作为目标
                            targets = [(center_x, center_y, center_z, center_range)]

                    if (
                        tracked_target_state.locked
                        and tracked_target_state.last_measurement_ms > 0
                        and (current_time - tracked_target_state.last_measurement_ms) > _TRACKER_TIMEOUT_MS
                    ):
                        tracked_target_state, tracker_event = tt.update_tracker(
                            tracked_target_state,
                            [],
                            current_time,
                            gate_m=_TRACKER_GATE_M,
                            max_miss=_TRACKER_MAX_MISS,
                            timeout_ms=_TRACKER_TIMEOUT_MS,
                            stable_hits=_TRACKER_STABLE_HITS,
                            relock_grace_ms=_TRACKER_RELOCK_GRACE_MS,
                            relock_distance_m=_TRACKER_RELOCK_DISTANCE_M,
                        )
                        if _tracker_event_requires_ml_reset(tracker_event, tracked_target_state):
                            height_history = []
                            _reset_ml_runtime_pipeline(reset_predictor=True)
                            last_fall_alert_result = _default_fall_alert_result('ML ready')
                        if tracker_event == 'predicted' and tracked_target_state.position is not None:
                            current_tracked_target = tracked_target_state.position
                            targets = [current_tracked_target]
                    
                    # -------------------------------------------------------
                    # 高度检测：只使用 tracked_target_state 主目标，与 detect_fall 共用同一份序列
                    # -------------------------------------------------------
                    if 'height_history' in globals() and 'height_plot' in globals() and len(x) > 0:
                        if current_tracked_target is not None:
                            try:
                                target_x, target_y, target_z, target_range = current_tracked_target
                                max_height_history = height_history_length.value()
                                updated_history = fd.update_height_history(
                                    height_history,
                                    (current_time, target_z, -target_y, target_range),
                                    max_height_history,
                                )
                                if len(updated_history) == len(height_history) and update_figure._debug_counter % 100 == 0:
                                    print(f"[高度检测] 过滤异常点: Z={target_z:.2f}m, Y={target_y:.2f}m, 距离={target_range:.2f}m")
                                height_history = updated_history

                                # 直接使用 height_history，不做第二轮 IQR 过滤
                                # 确保高度图与 detect_fall() 使用完全相同的序列
                                if len(height_history) > 0:
                                    height_z = [h[1] for h in height_history]
                                    height_y = [h[2] for h in height_history]
                                    height_ranges = [h[3] for h in height_history]

                                    # 更新高度散点图
                                    if len(height_ranges) > 0 and max(height_ranges) > min(height_ranges):
                                        height_colors = (np.array(height_ranges) - min(height_ranges)) / (max(height_ranges) - min(height_ranges))
                                        height_color_list = [(int(255*(1-c)), 0, int(255*c), 200) for c in height_colors]
                                    else:
                                        height_color_list = [(255, 0, 0, 200)] * len(height_z)

                                    height_plot.setData(x=height_y, y=height_z, brush=height_color_list, size=height_point_size.value())
                                    if len(height_y) > 1:
                                        height_line_ref.setData(x=height_y, y=height_z)
                                    else:
                                        height_line_ref.setData([], [])

                                    if len(height_z) > 0:
                                        cur_y_disp = -height_y[-1]
                                        cur_z_disp = height_z[-1]
                                        cur_r_disp = height_ranges[-1]
                                        height_y_orig = [-v for v in height_y]
                                        height_info = (
                                            f"高度点数: {len(height_z)} | "
                                            f"Z={cur_z_disp:.2f}m Y={cur_y_disp:.2f}m 距离={cur_r_disp:.2f}m | "
                                            f"Z范围: [{min(height_z):.2f}, {max(height_z):.2f}]m | "
                                            f"Y范围: [{min(height_y_orig):.2f}, {max(height_y_orig):.2f}]m"
                                        )
                                        height_info_label.setText(height_info)

                            except Exception as e:
                                print(f"高度检测更新错误: {e}")
                                import traceback
                                traceback.print_exc()
                else:
                    # 历史缓冲区为空，清空显示
                    try:
                        pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                                  color=np.empty((0, 4), dtype=np.float32), 
                                                  size=8)
                    except Exception as e:
                        print(f"清空点云错误: {e}")
                    pointcloud_info_label.setText("点云数量: 0 | 等待数据...")
                    if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                        pointcloud_coordinate_info_label.setText("等待点云数据...")
            except Exception as e:
                print(f"点云显示错误: {e}")
                import traceback
                traceback.print_exc()
                if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                    pointcloud_coordinate_info_label.setText("点云处理错误，请查看日志")
        else:
            # 历史缓冲区为空时，清空显示
            if len(PointCloudHistory) == 0:
                try:
                    pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                              color=np.empty((0, 4), dtype=np.float32), 
                                              size=8)
                except Exception as e:
                    print(f"清空点云错误: {e}")
            # 队列为空时也更新信息，显示队列状态
            if getattr(rtp, 'POINTCLOUD_DEBUG_LOGS', False) and update_figure._debug_counter % 200 == 0:
                print(f"[点云显示] 队列为空，等待数据...")
                print_pointcloud_log("队列状态: 等待数据...", fontcolor='orange')
            pointcloud_info_label.setText(f"点云数量: 0 | 队列状态: 等待数据...")
            if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                pointcloud_coordinate_info_label.setText("等待点云数据...")

    # 每轮刷新只消费一次最新热图帧，避免 GUI 线程被重复 queue.get() 卡住。
    _consume_ml_feature_frames()
    _update_realtime_detection_dashboard()
    _refresh_monitor_status_preview()

    QtCore.QTimer.singleShot(1, update_figure)

def printlog(string,fontcolor):
    logtxt.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    logtxt.append("<font color="+fontcolor+">"+str(gettime)+"-->"+string+"</font>")

def print_pointcloud_log(string, fontcolor='black'):
    """输出点云采集日志到点云日志文本框"""
    global pointcloud_log_text
    if pointcloud_log_text is not None:
        pointcloud_log_text.moveCursor(QtGui.QTextCursor.End)
        gettime = time.strftime("%H:%M:%S", time.localtime())
        pointcloud_log_text.append("<font color="+fontcolor+">"+str(gettime)+" --> "+string+"</font>")
        # 限制日志行数，避免内存占用过大
        if pointcloud_log_text.document().blockCount() > 500:
            cursor = pointcloud_log_text.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.movePosition(QtGui.QTextCursor.Down, QtGui.QTextCursor.MoveAnchor, 100)
            cursor.movePosition(QtGui.QTextCursor.StartOfLine)
            cursor.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
            cursor.removeSelectedText()

def getradarparameters():
    if radarparameters.currentIndex() > -1 and radarparameters.currentText() != '--select--':
        selected_cfg = radarparameters.currentText()
        radarparameters.setToolTip(selected_cfg)
        if capture_started and runtime_cfg is not None:
            active_file = runtime_cfg.get('config_file')
            if active_file and os.path.normpath(selected_cfg) != os.path.normpath(active_file):
                printlog('当前版本不支持采集中切换 cfg', fontcolor='orange')
                _update_runtime_ui(runtime_cfg)
                return
        apply_runtime_cfg(selected_cfg)

def openradar(config,com,data_port=None):
    global radar_ctrl, capture_started

    if capture_started:
        printlog('当前版本不支持运行中重新启动采集', fontcolor='orange')
        return False

    active_cfg = apply_runtime_cfg(config)
    if active_cfg is None:
        printlog('未找到有效的雷达配置文件', fontcolor='red')
        return False

    radar_ctrl = SerialConfig(name='ConnectRadar', CLIPort=com, BaudRate=115200)
    radar_ctrl.StopRadar()
    _build_runtime_workers()
    collector.start()
    processor.start()
    radar_ctrl.SendConfig(config)
    radar_ctrl.StartRadar()
    capture_started = True
    printlog('Radar started, collecting data', fontcolor='blue')
    print("Starting UDP listener and processor threads...")
    print("Current threads:", threading.enumerate())
    update_figure()
    _set_monitor_enabled(True)
    return True

def updatacomstatus(cbox):
    port_list = list(list_ports.comports())
    cbox.clear()
    for i in range(len(port_list)):
        cbox.addItem(str(port_list[i][0]))

def setserialport(cbox, com):
    global CLIport_name
    global Dataport_name
    if cbox.currentIndex() > -1:
        port = cbox.currentText()
        if com == "CLI":
            CLIport_name = port
   
        else:
            Dataport_name = port
    
def sendconfigfunc():
    global CLIport_name
    global Dataport_name
    if len(CLIport_name) != 0  and radarparameters.currentText() != '--select--':
        if openradar(radarparameters.currentText(), CLIport_name, Dataport_name):
            printlog(string = '发送成功', fontcolor='blue')
    else:
        printlog(string = '发送失败', fontcolor='red')


# 支持 Colormap切换用于图像热度色彩调整
def setcolor():
    if(color_.currentText()!='--select--' and color_.currentText()!=''):
        if color_.currentText() == 'customize':
            pgColormap = pg_get_cmap(color_.currentText())
        else:
            cmap=plt.cm.get_cmap(color_.currentText())
            pgColormap = pg_get_cmap(cmap)
        lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
        img_rdi.setLookupTable(lookup_table)
        img_rai.setLookupTable(lookup_table)
        img_rti.setLookupTable(lookup_table)
        img_dti.setLookupTable(lookup_table)
        img_rei.setLookupTable(lookup_table)

def get_filelist(dir,Filelist):
    newDir=dir
    #注意看dir是文件名还是路径＋文件名！！！！！！！！！！！！！！
    if os.path.isfile(dir):
        dir_ = os.path.basename(dir)  
        if (dir_[:2] == 'DT') and (dir_[-4:] == '.npy'):
            Filelist[0].append(dir)
        elif (dir_[:2] == 'RT') and (dir_[-4:] == '.npy'):
            Filelist[1].append(dir)
        elif (dir_[:3] == 'RDT') and (dir_[-4:] == '.npy'):
            Filelist[2].append(dir)
        elif (dir_[:3] == 'ART') and (dir_[-4:] == '.npy'):
            Filelist[3].append(dir)    
        elif (dir_[:3] == 'ERT') and (dir_[-4:] == '.npy'):
            Filelist[4].append(dir)  
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            get_filelist(newDir,Filelist)
    return Filelist

def show_sub():
    subWin.show()
    MainWindow.hide()



def application():
    global color_,radarparameters,maxVelocitylabel,maxRangelabel,dopplerResolutionlabel,rangeResolutionlabel,logtxt
    global monitor_button, fall_status_view
    global model_path_edit, model_status_label, replay_path_edit, replay_status_label, replay_start_button
    global realtime_dashboard_widget, realtime_dashboard_layout, realtime_prob_bar, realtime_confirm_bar
    global realtime_gate_label, realtime_summary_label, realtime_prob_curve, realtime_prob_scatter
    global realtime_model_status_label, ui
    global subWin,MainWindow
    global pointcloud_scatter, pointcloud_view, pointcloud_info_label, Dataportbox
    global pointcloud_max_points, pointcloud_refresh_rate, pointcloud_threshold
    global pointcloud_show_grid, pointcloud_show_axes
    global pointcloud_grid, pointcloud_grid_xy, pointcloud_grid_xz, pointcloud_grid_yz
    global pointcloud_x_axis, pointcloud_y_axis, pointcloud_z_axis
    global pointcloud_x_marker, pointcloud_y_marker, pointcloud_z_marker
    global pointcloud_last_update_time, pointcloud_refresh_interval
    global pointcloud_log_text, pointcloud_coordinate_info_label
    global trajectory_plot, trajectory_history, trajectory_view, trajectory_info_label, trajectory_plot_widget
    global trajectory_history_length, trajectory_point_size, trajectory_show_axes, trajectory_show_grid
    global target_count, trajectory_line_items, trajectory_scatter_items
    global trajectory_clear_button
    global height_plot, height_history, height_view, height_info_label, height_plot_widget
    global height_history_length, height_point_size, height_show_axes, height_show_grid
    global height_clear_button, height_line_ref
    global fall_alert_label, last_fall_alert_time, fall_alert_cooldown
    global fall_detection_enabled, fall_new_strategy_sensitivity
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()

    ui.setupUi(MainWindow)
    MainWindow.show()
    subWin = MiniStatusWindow(MainWindow)
    
    # 改了D:\Applications\anaconda3\Lib\site-packages\pyqtgraph\graphicsItems\ViewBox
    # 里的ViewBox.py第919行padding = self.suggestPadding(ax)改成padding = 0

    # 距离-多普勒图
    view_rdi = ui.graphicsView_6.addViewBox()
    ui.graphicsView_6.setCentralWidget(view_rdi)#去边界
    # 距离-方位角图
    view_rai = ui.graphicsView_4.addViewBox()
    ui.graphicsView_4.setCentralWidget(view_rai)#去边界
    # 时间-距离图
    # 绘制图像（2）：在ViewBox中创建一个画布view_rti
    view_rti = ui.graphicsView.addViewBox()
    ui.graphicsView.setCentralWidget(view_rti)#去边界
    # 多普勒-时间图
    view_dti = ui.graphicsView_2.addViewBox()
    ui.graphicsView_2.setCentralWidget(view_dti)#去边界
    # 距离-俯仰角图
    view_rei = ui.graphicsView_3.addViewBox()
    ui.graphicsView_3.setCentralWidget(view_rei)#去边界
    # 跌倒状态输出面板
    fall_status_view = ui.graphicsView_5
    fall_status_view.setAlignment(QtCore.Qt.AlignCenter)
    fall_status_view.setWordWrap(True)
    _set_fall_status_preview('Standby', level='idle')

    if not _realtime_heatmaps_enabled:
        ui.groupBox_9.setTitle("实时跌倒检测看板")
        ui.tabWidget.setTabText(ui.tabWidget.indexOf(ui.tab), "实时检测")

        dashboard_hidden_widgets = [
            ui.graphicsView,
            ui.graphicsView_2,
            ui.graphicsView_3,
            ui.graphicsView_4,
            ui.graphicsView_5,
            ui.graphicsView_6,
            ui.label,
            ui.label_7,
            ui.label_8,
            ui.label_9,
            ui.label_10,
            ui.label_11,
        ]
        for widget in dashboard_hidden_widgets:
            widget.hide()

        realtime_dashboard_scroll = QtWidgets.QScrollArea(ui.groupBox_9)
        realtime_dashboard_scroll.setWidgetResizable(True)
        realtime_dashboard_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        realtime_dashboard_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        realtime_dashboard_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        realtime_dashboard_root = QtWidgets.QWidget(realtime_dashboard_scroll)
        realtime_dashboard_root_layout = QtWidgets.QVBoxLayout(realtime_dashboard_root)
        realtime_dashboard_root_layout.setContentsMargins(12, 12, 12, 12)
        realtime_dashboard_root_layout.setSpacing(12)

        realtime_dashboard_widget = QtWidgets.QWidget(realtime_dashboard_root)
        realtime_dashboard_layout = QtWidgets.QVBoxLayout(realtime_dashboard_widget)
        realtime_dashboard_layout.setContentsMargins(4, 4, 4, 4)
        realtime_dashboard_layout.setSpacing(10)

        dashboard_header = QtWidgets.QLabel("Realtime Fall Detection")
        dashboard_header.setAlignment(QtCore.Qt.AlignCenter)
        dashboard_header.setStyleSheet(
            "font-size: 26px;"
            "font-weight: bold;"
            "color: #1f2d3d;"
            "padding: 4px;"
        )
        realtime_dashboard_layout.addWidget(dashboard_header)

        fall_status_view = QtWidgets.QLabel(realtime_dashboard_widget)
        fall_status_view.setMinimumHeight(120)
        fall_status_view.setMaximumHeight(150)
        fall_status_view.setWordWrap(True)
        fall_status_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        realtime_dashboard_layout.addWidget(fall_status_view)

        progress_widget = QtWidgets.QWidget(realtime_dashboard_widget)
        progress_layout = QtWidgets.QGridLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setHorizontalSpacing(12)
        progress_layout.setVerticalSpacing(8)

        gate_title = QtWidgets.QLabel("Target Gate")
        prob_title = QtWidgets.QLabel("Fall Probability")
        confirm_title = QtWidgets.QLabel("Confirmation Progress")
        for label_widget in (gate_title, prob_title, confirm_title):
            label_widget.setStyleSheet("font-size: 14px; font-weight: bold; color: #3c4858;")

        realtime_gate_label = QtWidgets.QLabel(progress_widget)
        realtime_prob_bar = QtWidgets.QProgressBar(progress_widget)
        realtime_confirm_bar = QtWidgets.QProgressBar(progress_widget)
        realtime_prob_bar.setRange(0, 100)
        realtime_confirm_bar.setRange(0, 2)
        realtime_prob_bar.setValue(0)
        realtime_confirm_bar.setValue(0)
        realtime_prob_bar.setTextVisible(True)
        realtime_confirm_bar.setTextVisible(True)
        realtime_prob_bar.setStyleSheet(
            "QProgressBar {border: 1px solid #b0bec5; border-radius: 6px; background: #f4f7fb; text-align: center; min-height: 24px;}"
            "QProgressBar::chunk {background-color: #d0021b; border-radius: 6px;}"
        )
        realtime_confirm_bar.setStyleSheet(
            "QProgressBar {border: 1px solid #b0bec5; border-radius: 6px; background: #f4f7fb; text-align: center; min-height: 24px;}"
            "QProgressBar::chunk {background-color: #4a90e2; border-radius: 6px;}"
        )

        progress_layout.addWidget(gate_title, 0, 0)
        progress_layout.addWidget(prob_title, 0, 1)
        progress_layout.addWidget(confirm_title, 2, 0)
        progress_layout.addWidget(realtime_gate_label, 1, 0)
        progress_layout.addWidget(realtime_prob_bar, 1, 1)
        progress_layout.addWidget(realtime_confirm_bar, 3, 0, 1, 2)
        progress_layout.setColumnStretch(0, 1)
        progress_layout.setColumnStretch(1, 2)
        realtime_dashboard_layout.addWidget(progress_widget)

        realtime_summary_label = QtWidgets.QPlainTextEdit(realtime_dashboard_widget)
        realtime_summary_label.setReadOnly(True)
        realtime_summary_label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        realtime_summary_label.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        realtime_summary_label.setMinimumHeight(120)
        realtime_summary_label.setMaximumHeight(160)
        realtime_summary_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        realtime_summary_label.setStyleSheet(
            "border-width: 1px;"
            "border-style: solid;"
            "border-color: #d0d7de;"
            "background-color: #ffffff;"
            "color: #2f3b52;"
            "font-size: 13px;"
            "padding: 6px;"
        )
        realtime_dashboard_layout.addWidget(realtime_summary_label)
        realtime_dashboard_root_layout.addWidget(realtime_dashboard_widget, 0)

        realtime_prob_plot_widget = pg.PlotWidget(realtime_dashboard_root)
        realtime_prob_plot_widget.setBackground('#ffffff')
        realtime_prob_plot_widget.setMinimumHeight(220)
        realtime_prob_plot_widget.setMaximumHeight(260)
        realtime_prob_plot_widget.showGrid(x=True, y=True, alpha=0.25)
        realtime_prob_plot_widget.setTitle("Recent Inference Probability")
        realtime_prob_plot_widget.setLabel('left', 'fall probability')
        realtime_prob_plot_widget.setLabel('bottom', 'frame index')
        realtime_prob_plot_widget.setYRange(0.0, 1.0, padding=0.02)
        realtime_prob_curve = realtime_prob_plot_widget.plot(
            [],
            [],
            pen=pg.mkPen('#d0021b', width=3),
        )
        realtime_prob_scatter = pg.ScatterPlotItem()
        realtime_prob_plot_widget.addItem(realtime_prob_scatter)
        realtime_dashboard_root_layout.addWidget(realtime_prob_plot_widget, 0)
        realtime_dashboard_root_layout.addStretch(1)

        realtime_dashboard_scroll.setWidget(realtime_dashboard_root)
        ui.gridLayout.addWidget(realtime_dashboard_scroll, 0, 0, 4, 3)
        _set_fall_status_preview('Standby', level='idle')
        _update_realtime_detection_dashboard()


    sendcfgbtn = ui.pushButton_11
    exitbtn = ui.pushButton_12
    monitor_button = ui.pushButton_15

    color_ = ui.comboBox
    radarparameters = ui.comboBox_7
    Cliportbox = ui.comboBox_8
    Dataportbox = ui.comboBox_10

    logtxt = ui.textEdit
    changepage = ui.actionload

    model_group_box = QtWidgets.QGroupBox('模型推理')
    model_layout = QtWidgets.QVBoxLayout(model_group_box)
    model_path_row = QtWidgets.QHBoxLayout()
    model_path_edit = QtWidgets.QLineEdit(model_group_box)
    model_path_edit.setPlaceholderText('选择 .pth / .py / .pt / .jit / .ts 模型文件')
    model_browse_button = QtWidgets.QPushButton('browse', model_group_box)
    model_path_row.addWidget(model_path_edit)
    model_path_row.addWidget(model_browse_button)
    model_layout.addLayout(model_path_row)
    model_load_button = QtWidgets.QPushButton('load model', model_group_box)
    model_layout.addWidget(model_load_button)
    model_status_label = QtWidgets.QLabel(model_group_box)
    _set_runtime_aux_status(
        model_status_label,
        _ml_load_message,
        level='idle',
    )
    model_layout.addWidget(model_status_label)
    ui.verticalLayout_7.addWidget(model_group_box)

    replay_group_box = QtWidgets.QGroupBox('数据回放')
    replay_layout = QtWidgets.QVBoxLayout(replay_group_box)
    replay_path_row = QtWidgets.QHBoxLayout()
    replay_path_edit = QtWidgets.QLineEdit(replay_group_box)
    replay_path_edit.setPlaceholderText('预留 .bin 回放文件入口')
    replay_browse_button = QtWidgets.QPushButton('browse', replay_group_box)
    replay_path_row.addWidget(replay_path_edit)
    replay_path_row.addWidget(replay_browse_button)
    replay_layout.addLayout(replay_path_row)
    replay_load_button = QtWidgets.QPushButton('load replay', replay_group_box)
    replay_layout.addWidget(replay_load_button)
    replay_start_button = QtWidgets.QPushButton('replay reserved', replay_group_box)
    replay_start_button.setEnabled(False)
    replay_layout.addWidget(replay_start_button)
    replay_status_label = QtWidgets.QLabel(replay_group_box)
    _set_runtime_aux_status(
        replay_status_label,
        'Replay interface reserved for offline .bin playback. Parser wiring is pending.',
        level='idle',
    )
    replay_layout.addWidget(replay_status_label)
    ui.verticalLayout_7.addWidget(replay_group_box)
    

    rangeResolutionlabel = ui.label_14
    dopplerResolutionlabel = ui.label_35
    maxRangelabel = ui.label_16
    maxVelocitylabel = ui.label_37
    _populate_cfg_selector()
    config_params = apply_runtime_cfg() or {}

    def _browse_model_file():
        selected_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            MainWindow,
            '选择模型文件',
            '',
            'Predictor Files (*.py *.pt *.jit *.ts *.pth);;All Files (*)',
        )
        if selected_file:
            model_path_edit.setText(selected_file)

    def _load_model_file():
        global _fall_predictor, _loaded_model_path

        model_path = model_path_edit.text().strip()
        result = fp.load_predictor_from_path(model_path, runtime_cfg=runtime_cfg)
        if result.success:
            _fall_predictor = result.predictor
            _loaded_model_path = result.model_path
            _refresh_ml_contract_state(
                load_message=result.message,
                contract=result.contract,
                validation=result.contract_validation,
            )
            _reset_ml_runtime_pipeline(reset_predictor=True)
            if result.contract_validation is not None and not result.contract_validation.valid:
                printlog(
                    f"{result.message} | {result.contract_validation.message}",
                    fontcolor='orange',
                )
            else:
                printlog(result.message, fontcolor='blue')
        else:
            _fall_predictor = fp.build_fall_predictor()
            _loaded_model_path = ""
            _refresh_ml_contract_state(load_message=result.message)
            _reset_ml_runtime_pipeline(reset_predictor=False)
            printlog(result.message, fontcolor='orange')

    def _autoload_default_model():
        default_model_path = fp.discover_default_model_path()
        if not default_model_path:
            return
        if model_path_edit.text().strip():
            return
        model_path_edit.setText(default_model_path)
        _load_model_file()

    def _browse_replay_file():
        selected_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            MainWindow,
            '选择回放文件',
            '',
            'Radar Binary (*.bin);;All Files (*)',
        )
        if selected_file:
            replay_path_edit.setText(selected_file)

    def _load_replay_file():
        global _replay_source, _loaded_replay_path

        replay_path = replay_path_edit.text().strip()
        result = rc.load_replay_source(replay_path, runtime_cfg=runtime_cfg)
        if result.success:
            _replay_source = result.source
            _loaded_replay_path = result.replay_path
            _set_runtime_aux_status(replay_status_label, result.message, level='ready')
            printlog(result.message, fontcolor='blue')
        else:
            _set_runtime_aux_status(replay_status_label, result.message, level='warn')
            printlog(result.message, fontcolor='orange')

    model_browse_button.clicked.connect(_browse_model_file)
    model_load_button.clicked.connect(_load_model_file)
    replay_browse_button.clicked.connect(_browse_replay_file)
    replay_load_button.clicked.connect(_load_replay_file)
    _autoload_default_model()
    
    # 点云显示相关
    pointcloud_view = ui.graphicsView_pointcloud
    pointcloud_info_label = ui.label_pointcloud_info
    pointcloud_log_text = ui.textEdit_pointcloud_log
    pointcloud_coordinate_info_label = ui.label_coordinate_info
    
    # 点云配置控件
    pointcloud_max_points = ui.spinBox_max_points
    pointcloud_refresh_rate = ui.spinBox_refresh_rate
    pointcloud_threshold = ui.doubleSpinBox_threshold
    pointcloud_show_grid = ui.checkBox_show_grid
    pointcloud_show_axes = ui.checkBox_show_axes
    pointcloud_test_button = ui.pushButton_test_pointcloud
    pointcloud_enable_clustering = ui.checkBox_enable_clustering
    pointcloud_cluster_eps = ui.doubleSpinBox_cluster_eps
    pointcloud_cluster_min_samples = ui.spinBox_cluster_min_samples
    
    # 设置默认参数值（根据用户界面配置）
    pointcloud_max_points.setValue(1000)  # 最大点云数量: 1000
    pointcloud_refresh_rate.setValue(20)  # 刷新速率: 20ms
    pointcloud_threshold.setValue(0.45)  # 检测阈值比例: 0.45
    pointcloud_show_grid.setChecked(True)  # 显示网格: 已勾选
    pointcloud_show_axes.setChecked(True)  # 显示坐标系: 已勾选
    pointcloud_enable_clustering.setChecked(False)  # 显示聚类后的点云: 未勾选
    pointcloud_cluster_eps.setValue(0.15)  # 聚类距离阈值: 0.15m
    pointcloud_cluster_min_samples.setValue(5)  # 最小聚类点数: 5
    
    # 初始化阈值参数到全局变量
    gl.set_value('pointcloud_threshold', pointcloud_threshold.value())
    
    # 测试点云按钮功能
    def generate_test_pointcloud():
        """生成测试点云用于调试"""
        # 生成一些测试点：一个立方体形状的点云
        test_points = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    range_val = np.sqrt(x**2 + y**2 + z**2)
                    if range_val > 0.1:  # 排除原点
                        test_points.append([range_val, x, y, z])
        
        test_pointcloud = np.array(test_points, dtype=np.float32)
        # 清空队列并放入测试数据
        while not PointCloudData.empty():
            try:
                PointCloudData.get_nowait()
            except:
                break
        PointCloudData.put(test_pointcloud)
        printlog(f'生成测试点云: {len(test_pointcloud)} 个点', fontcolor='green')
        print_pointcloud_log(f'测试点云已生成: {len(test_pointcloud)} 个点', fontcolor='green')
    
    pointcloud_test_button.clicked.connect(generate_test_pointcloud)
    
    # 配置3D点云显示（pointcloud_view已经是GLViewWidget）
    pointcloud_view.setCameraPosition(distance=10, elevation=30, azimuth=45)
    # 改善背景色：使用深灰色而不是纯黑色，更容易看清
    pointcloud_view.setBackgroundColor('#1e1e1e')  # 深灰色背景
    
    # 坐标系原点设置为(-1, -1, -1)
    origin_x, origin_y, origin_z = -1, -1, -1
    axis_length = 10  # 坐标轴长度
    
    # 创建三个平面的网格（每个平面都有网格，使用相同的样式）
    grid_color = (120, 120, 120, 150)  # 统一的网格颜色和透明度
    grid_spacing = 1  # 网格间距（米）
    
    # XY平面网格（Z=-1平面，水平面）
    grid_xy = GLGridItem()
    grid_xy.setSize(x=axis_length, y=axis_length, z=0)  # XY平面，z=0
    grid_xy.setSpacing(x=grid_spacing, y=grid_spacing, z=grid_spacing)
    grid_xy.setColor(grid_color)
    grid_xy.translate(origin_x + axis_length/2, origin_y + axis_length/2, origin_z)  # 移动到Z=-1平面
    pointcloud_view.addItem(grid_xy)
    
    # XZ平面网格（Y=-1平面，侧视图）
    grid_xz = GLGridItem()
    grid_xz.setSize(x=axis_length, y=0, z=axis_length)  # XZ平面，y=0
    grid_xz.setSpacing(x=grid_spacing, y=grid_spacing, z=grid_spacing)
    grid_xz.setColor(grid_color)  # 使用和XY平面相同的颜色
    grid_xz.rotate(90, 1, 0, 0)  # 绕X轴旋转90度
    grid_xz.translate(origin_x + axis_length/2, origin_y, origin_z + axis_length/2)  # 移动到Y=-1平面
    pointcloud_view.addItem(grid_xz)
    
    # YZ平面网格（X=-1平面，前视图）
    grid_yz = GLGridItem()
    grid_yz.setSize(x=0, y=axis_length, z=axis_length)  # YZ平面，x=0
    grid_yz.setSpacing(x=grid_spacing, y=grid_spacing, z=grid_spacing)
    grid_yz.setColor(grid_color)  # 使用和XY平面相同的颜色
    grid_yz.rotate(90, 0, 1, 0)  # 绕Y轴旋转90度
    grid_yz.translate(origin_x, origin_y + axis_length/2, origin_z + axis_length/2)  # 移动到X=-1平面
    pointcloud_view.addItem(grid_yz)
    
    # 创建坐标系（X, Y, Z轴），从原点(-5, -5, -5)开始
    # X轴 - 红色（前方）
    x_axis = GLLinePlotItem()
    x_axis.setData(pos=np.array([[origin_x, origin_y, origin_z], 
                                 [origin_x + axis_length, origin_y, origin_z]]), 
                   color=(1, 0, 0, 1), width=5)
    pointcloud_view.addItem(x_axis)
    
    # Y轴 - 绿色（左侧）
    y_axis = GLLinePlotItem()
    y_axis.setData(pos=np.array([[origin_x, origin_y, origin_z], 
                                 [origin_x, origin_y + axis_length, origin_z]]), 
                   color=(0, 1, 0, 1), width=5)
    pointcloud_view.addItem(y_axis)
    
    # Z轴 - 蓝色（上方）
    z_axis = GLLinePlotItem()
    z_axis.setData(pos=np.array([[origin_x, origin_y, origin_z], 
                                 [origin_x, origin_y, origin_z + axis_length]]), 
                   color=(0, 0, 1, 1), width=5)
    pointcloud_view.addItem(z_axis)
    
    # 在坐标轴末端添加箭头标记
    arrow_size = 0.3
    # X轴末端标记（前方）- 红色箭头
    x_arrow_points = np.array([
        [origin_x + axis_length, origin_y, origin_z],  # 箭头尖端
        [origin_x + axis_length - arrow_size, origin_y + arrow_size*0.2, origin_z],  # 箭头左翼
        [origin_x + axis_length - arrow_size, origin_y - arrow_size*0.2, origin_z], # 箭头右翼
    ], dtype=np.float32)
    x_marker = GLScatterPlotItem()
    x_marker.setData(pos=x_arrow_points, color=np.array([(1, 0, 0, 1)] * len(x_arrow_points), dtype=np.float32), size=12)
    pointcloud_view.addItem(x_marker)
    
    # Y轴末端标记（左侧）- 绿色箭头
    y_arrow_points = np.array([
        [origin_x, origin_y + axis_length, origin_z],  # 箭头尖端
        [origin_x - arrow_size*0.2, origin_y + axis_length - arrow_size, origin_z], # 箭头左翼
        [origin_x + arrow_size*0.2, origin_y + axis_length - arrow_size, origin_z],  # 箭头右翼
    ], dtype=np.float32)
    y_marker = GLScatterPlotItem()
    y_marker.setData(pos=y_arrow_points, color=np.array([(0, 1, 0, 1)] * len(y_arrow_points), dtype=np.float32), size=12)
    pointcloud_view.addItem(y_marker)
    
    # Z轴末端标记（上方）- 蓝色箭头
    z_arrow_points = np.array([
        [origin_x, origin_y, origin_z + axis_length],  # 箭头尖端
        [origin_x + arrow_size*0.2, origin_y, origin_z + axis_length - arrow_size],  # 箭头左翼
        [origin_x - arrow_size*0.2, origin_y, origin_z + axis_length - arrow_size], # 箭头右翼
    ], dtype=np.float32)
    z_marker = GLScatterPlotItem()
    z_marker.setData(pos=z_arrow_points, color=np.array([(0, 0, 1, 1)] * len(z_arrow_points), dtype=np.float32), size=12)
    pointcloud_view.addItem(z_marker)
    
    # 保存坐标系和网格引用，以便控制显示/隐藏
    pointcloud_grid = grid_xy  # 主网格（XY平面）
    pointcloud_grid_xy = grid_xy  # XY平面网格
    pointcloud_grid_xz = grid_xz  # XZ平面网格
    pointcloud_grid_yz = grid_yz  # YZ平面网格
    pointcloud_x_axis = x_axis
    pointcloud_y_axis = y_axis
    pointcloud_z_axis = z_axis
    pointcloud_x_marker = x_marker
    pointcloud_y_marker = y_marker
    pointcloud_z_marker = z_marker
    
    # 创建散点图项
    pointcloud_scatter = GLScatterPlotItem()
    pointcloud_scatter.setGLOptions('opaque')
    pointcloud_view.addItem(pointcloud_scatter)
    
    # 点云刷新速率控制变量
    pointcloud_last_update_time = 0
    pointcloud_refresh_interval = 50  # 默认50ms
    
    # 轨迹显示相关
    trajectory_view_container = ui.graphicsView_trajectory
    trajectory_info_label = ui.label_trajectory_info
    trajectory_history_length = ui.spinBox_trajectory_history
    trajectory_point_size = ui.spinBox_trajectory_point_size
    trajectory_show_axes = ui.checkBox_trajectory_show_axes
    trajectory_show_grid = ui.checkBox_trajectory_show_grid
    target_count = ui.spinBox_target_count
    trajectory_clear_button = ui.pushButton_clear_trajectory
    
    # 初始化轨迹历史存储（存储格式: [(timestamp, target_id, x, y, range), ...]）
    trajectory_history = []
    
    # 存储轨迹连线对象（用于多目标显示）
    global trajectory_line_items, trajectory_scatter_items
    trajectory_line_items = []
    trajectory_scatter_items = []
    
    # 雷达参数来自当前cfg
    MAX_RANGE = float(config_params.get('maxRange', 5.4))
    RANGE_RESOLUTION = float(config_params.get('range_resolution_m', config_params.get('rangeResolutionMeters', 0.09)))
    
    # 配置轨迹2D显示（XY平面，类似高度检测页面的显示方式）
    trajectory_plot_widget = trajectory_view_container.addPlot(title=f"目标运动轨迹 (XY平面) | 距离分辨率: {RANGE_RESOLUTION*100:.1f}cm | 范围: ±{MAX_RANGE:.1f}m")
    trajectory_plot_widget.setLabel('left', 'Y (m)', color='black', size='12pt')  # Y轴（垂直）
    trajectory_plot_widget.setLabel('bottom', 'X (m)', color='black', size='12pt')  # X轴（水平）
    trajectory_plot_widget.showGrid(x=True, y=True, alpha=0.3)
    trajectory_plot_widget.setAspectLocked(True)  # 锁定纵横比
    # 设置显示范围
    trajectory_plot_widget.setXRange(-MAX_RANGE, MAX_RANGE)  # X轴范围
    trajectory_plot_widget.setYRange(-MAX_RANGE, MAX_RANGE)  # Y轴范围
    
    # 绘制坐标轴（X轴水平，Y轴垂直）
    # X轴（红色，水平）
    x_axis_line_traj = pg.PlotDataItem([-MAX_RANGE, MAX_RANGE], [0, 0], pen=pg.mkPen(color='r', width=2), name='X轴')
    trajectory_plot_widget.addItem(x_axis_line_traj)
    # Y轴（绿色，垂直）
    y_axis_line_traj = pg.PlotDataItem([0, 0], [-MAX_RANGE, MAX_RANGE], pen=pg.mkPen(color='g', width=2), name='Y轴')
    trajectory_plot_widget.addItem(y_axis_line_traj)
    
    # 标注原点
    origin_text_traj = pg.TextItem('雷达原点 (0,0)', anchor=(0.5, 1.5), color='blue')
    origin_text_traj.setPos(0, 0)
    # 设置字体大小
    font_traj = QtGui.QFont()
    font_traj.setPointSize(10)
    origin_text_traj.setFont(font_traj)
    trajectory_plot_widget.addItem(origin_text_traj)
    
    # 标注X轴方向（水平，右侧）
    x_axis_text_traj = pg.TextItem('X轴 (前方)', anchor=(0.5, 0.5), color='red')
    x_axis_text_traj.setPos(MAX_RANGE * 0.7, 0.2)  # 放在右侧
    x_axis_text_traj.setFont(font_traj)
    trajectory_plot_widget.addItem(x_axis_text_traj)
    
    # 标注Y轴方向（垂直，上方）
    y_axis_text_traj = pg.TextItem('Y轴 (右侧)', anchor=(0.5, 0.5), color='green')
    y_axis_text_traj.setPos(0.2, MAX_RANGE * 0.7)  # 放在上方
    y_axis_text_traj.setFont(font_traj)
    trajectory_plot_widget.addItem(y_axis_text_traj)
    
    # 轨迹散点图（2D）
    trajectory_plot = pg.ScatterPlotItem(size=trajectory_point_size.value(), pen=pg.mkPen(width=1), brush=pg.mkBrush(255, 0, 0, 200))
    trajectory_plot_widget.addItem(trajectory_plot)
    
    # 轨迹连线（可选，用于显示轨迹路径）
    trajectory_line_ref = pg.PlotDataItem(pen=pg.mkPen(color='blue', width=2, style=QtCore.Qt.DashLine), name='轨迹连线')
    trajectory_plot_widget.addItem(trajectory_line_ref)
    
    # 保留trajectory_view_container引用，用于兼容性
    trajectory_view = trajectory_view_container
    
    # 清除轨迹按钮功能
    def clear_trajectory():
        global trajectory_history, trajectory_line_items, trajectory_scatter_items
        trajectory_history = []
        trajectory_plot.setData([], [])
        trajectory_line_ref.setData([], [])
        # 清除所有连线和散点图
        for line_item in trajectory_line_items:
            trajectory_plot_widget.removeItem(line_item)
        for scatter_item in trajectory_scatter_items:
            trajectory_plot_widget.removeItem(scatter_item)
        trajectory_line_items = []
        trajectory_scatter_items = []
        trajectory_info_label.setText("轨迹信息: 已清除")
    
    trajectory_clear_button.clicked.connect(clear_trajectory)
    
    # 控制网格和坐标轴的显示
    def update_trajectory_display():
        if trajectory_show_grid.isChecked():
            trajectory_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        else:
            trajectory_plot_widget.showGrid(x=False, y=False)
        
        if trajectory_show_axes.isChecked():
            x_axis_line_traj.show()
            y_axis_line_traj.show()
            origin_text_traj.show()
            x_axis_text_traj.show()
            y_axis_text_traj.show()
        else:
            x_axis_line_traj.hide()
            y_axis_line_traj.hide()
            origin_text_traj.hide()
            x_axis_text_traj.hide()
            y_axis_text_traj.hide()
    
    trajectory_show_grid.stateChanged.connect(lambda: update_trajectory_display())
    trajectory_show_axes.stateChanged.connect(lambda: update_trajectory_display())
    
    # 高度检测显示相关
    height_view = ui.graphicsView_height
    height_info_label = ui.label_height_info
    height_history_length = ui.spinBox_height_history
    height_point_size = ui.spinBox_height_point_size
    height_show_axes = ui.checkBox_height_show_axes
    height_show_grid = ui.checkBox_height_show_grid
    height_clear_button = ui.pushButton_clear_height
    fall_detection_enabled = ui.checkBox_enable_fall_detection  # 跌倒检测开关
    fall_new_strategy_sensitivity = ui.comboBox_new_strategy_sensitivity  # 新策略灵敏度等级
    fall_alert_label = ui.label_fall_alert  # 跌倒提示标签
    
    # 初始化高度历史存储（存储格式: [(timestamp, z, y, range), ...]）
    height_history = []
    
    # 初始化跌倒检测相关变量
    last_fall_alert_time = 0  # 上次跌倒提示时间，避免重复提示
    fall_alert_cooldown = 3000  # 跌倒提示冷却时间（毫秒），3秒内不重复提示

    legacy_height_widgets = [
        ui.checkBox_enable_new_fall_strategy,
        ui.doubleSpinBox_fall_height_min,
        ui.doubleSpinBox_fall_height_max,
    ]
    for widget in legacy_height_widgets:
        widget.setVisible(False)

    legacy_height_labels = [
        'label_fall_height_min',
        'label_fall_height_max',
        'label_new_fall_strategy',
    ]
    for label_name in legacy_height_labels:
        if hasattr(ui, label_name):
            getattr(ui, label_name).setVisible(False)

    fall_detection_enabled.setText("启用 ML 跌倒判定")
    if hasattr(ui, 'label_new_strategy_sensitivity'):
        ui.label_new_strategy_sensitivity.setText("ML 预设档位:")
    fall_new_strategy_sensitivity.clear()
    fall_new_strategy_sensitivity.addItems(["灵敏", "中等", "稳健"])
    fall_new_strategy_sensitivity.setCurrentText("中等")

    if hasattr(ui, 'label_fall_time_window'):
        ui.label_fall_time_window.setVisible(True)
        ui.label_fall_time_window.setText("连续阳性次数:")
    ui.doubleSpinBox_fall_time_window.setVisible(True)
    ui.doubleSpinBox_fall_time_window.setDecimals(0)
    ui.doubleSpinBox_fall_time_window.setMinimum(1.0)
    ui.doubleSpinBox_fall_time_window.setMaximum(5.0)
    ui.doubleSpinBox_fall_time_window.setSingleStep(1.0)
    ui.doubleSpinBox_fall_time_window.setValue(2.0)

    if hasattr(ui, 'label_fall_height_threshold'):
        ui.label_fall_height_threshold.setVisible(True)
        ui.label_fall_height_threshold.setText("ML 跌倒阈值:")
    ui.doubleSpinBox_fall_height_threshold.setVisible(True)
    ui.doubleSpinBox_fall_height_threshold.setDecimals(2)
    ui.doubleSpinBox_fall_height_threshold.setMinimum(0.50)
    ui.doubleSpinBox_fall_height_threshold.setMaximum(0.99)
    ui.doubleSpinBox_fall_height_threshold.setSingleStep(0.05)
    ui.doubleSpinBox_fall_height_threshold.setValue(0.50)

    def _apply_ml_preset(preset_name):
        preset = mla.get_preset_config(preset_name)
        ui.doubleSpinBox_fall_height_threshold.blockSignals(True)
        ui.doubleSpinBox_fall_time_window.blockSignals(True)
        ui.doubleSpinBox_fall_height_threshold.setValue(float(preset['threshold']))
        ui.doubleSpinBox_fall_time_window.setValue(float(preset['required_streak']))
        ui.doubleSpinBox_fall_height_threshold.blockSignals(False)
        ui.doubleSpinBox_fall_time_window.blockSignals(False)
        _sync_ml_alarm_state(_last_ml_prediction)
        _update_model_status_label()

    def _on_ml_setting_changed(*_args):
        _sync_ml_alarm_state(_last_ml_prediction)
        _update_model_status_label()

    # 灵敏度选择始终跟随跌倒检测主开关
    def on_fall_detection_toggled(enabled):
        enabled = bool(enabled)
        fall_new_strategy_sensitivity.setEnabled(enabled)
        ui.doubleSpinBox_fall_time_window.setEnabled(enabled)
        ui.doubleSpinBox_fall_height_threshold.setEnabled(enabled)
        if 'label_new_strategy_sensitivity' in dir(ui):
            ui.label_new_strategy_sensitivity.setEnabled(enabled)
        if hasattr(ui, 'label_fall_time_window'):
            ui.label_fall_time_window.setEnabled(enabled)
        if hasattr(ui, 'label_fall_height_threshold'):
            ui.label_fall_height_threshold.setEnabled(enabled)
        if not enabled and fall_alert_label is not None:
            fall_alert_label.hide()
        if not enabled:
            _reset_ml_runtime_pipeline(reset_predictor=True)
        else:
            _sync_ml_alarm_state(_last_ml_prediction)
            _update_model_status_label()

    fall_new_strategy_sensitivity.currentTextChanged.connect(_apply_ml_preset)
    ui.doubleSpinBox_fall_time_window.valueChanged.connect(_on_ml_setting_changed)
    ui.doubleSpinBox_fall_height_threshold.valueChanged.connect(_on_ml_setting_changed)
    fall_detection_enabled.stateChanged.connect(on_fall_detection_toggled)
    _apply_ml_preset(fall_new_strategy_sensitivity.currentText())
    on_fall_detection_toggled(fall_detection_enabled.isChecked())

    if 'realtime_dashboard_layout' in globals() and realtime_dashboard_layout is not None:
        if hasattr(ui, 'groupBox_3'):
            ui.groupBox_3.hide()
        if hasattr(ui, 'label_fall_detection'):
            ui.label_fall_detection.hide()

        realtime_controls_group = QtWidgets.QGroupBox('检测控制', realtime_dashboard_widget)
        realtime_controls_layout = QtWidgets.QGridLayout(realtime_controls_group)
        realtime_controls_layout.setContentsMargins(10, 10, 10, 10)
        realtime_controls_layout.setHorizontalSpacing(12)
        realtime_controls_layout.setVerticalSpacing(8)
        realtime_controls_group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        monitor_button.setParent(realtime_controls_group)
        fall_detection_enabled.setParent(realtime_controls_group)
        ui.label_new_strategy_sensitivity.setParent(realtime_controls_group)
        fall_new_strategy_sensitivity.setParent(realtime_controls_group)
        ui.label_fall_time_window.setParent(realtime_controls_group)
        ui.doubleSpinBox_fall_time_window.setParent(realtime_controls_group)
        ui.label_fall_height_threshold.setParent(realtime_controls_group)
        ui.doubleSpinBox_fall_height_threshold.setParent(realtime_controls_group)

        realtime_controls_layout.addWidget(monitor_button, 0, 0, 1, 2)
        realtime_controls_layout.addWidget(fall_detection_enabled, 1, 0, 1, 2)
        realtime_controls_layout.addWidget(ui.label_new_strategy_sensitivity, 2, 0)
        realtime_controls_layout.addWidget(fall_new_strategy_sensitivity, 2, 1)
        realtime_controls_layout.addWidget(ui.label_fall_time_window, 3, 0)
        realtime_controls_layout.addWidget(ui.doubleSpinBox_fall_time_window, 3, 1)
        realtime_controls_layout.addWidget(ui.label_fall_height_threshold, 4, 0)
        realtime_controls_layout.addWidget(ui.doubleSpinBox_fall_height_threshold, 4, 1)

        realtime_model_status_label = QtWidgets.QLabel(realtime_controls_group)
        realtime_model_status_label.setWordWrap(True)
        realtime_model_status_label.setMaximumHeight(96)
        _set_runtime_aux_status(
            realtime_model_status_label,
            _ml_load_message,
            level='idle',
        )
        realtime_controls_layout.addWidget(realtime_model_status_label, 5, 0, 1, 2)
        realtime_dashboard_layout.insertWidget(1, realtime_controls_group)
        _update_model_status_label()
    
    # 配置高度2D显示（Z-Y轴，Z轴朝上）
    # 高度范围：-1m到1m
    HEIGHT_MAX_RANGE = 1.0  # 高度最大范围1米
    height_plot_widget = height_view.addPlot(title=f"高度检测 (ZY平面) | 距离分辨率: {RANGE_RESOLUTION*100:.1f}cm | 高度范围: ±{HEIGHT_MAX_RANGE:.1f}m")
    height_plot_widget.setLabel('left', 'Z (m)', color='black', size='12pt')  # Z轴朝上（垂直）
    height_plot_widget.setLabel('bottom', 'Y (m)', color='black', size='12pt')  # Y轴水平
    height_plot_widget.showGrid(x=True, y=True, alpha=0.3)
    height_plot_widget.setAspectLocked(True)  # 锁定纵横比
    # 设置显示范围（高度范围-1m到1m，Y轴保持原范围）
    height_plot_widget.setXRange(-MAX_RANGE, MAX_RANGE)  # X轴显示Y值（水平，保持原范围）
    height_plot_widget.setYRange(-HEIGHT_MAX_RANGE, HEIGHT_MAX_RANGE)  # Y轴显示Z值（垂直，朝上，-1m到1m）
    
    # 绘制坐标轴（Z轴朝上，Y轴水平）
    # Y轴（绿色，水平）- X轴位置显示Y值
    y_axis_line_height = pg.PlotDataItem([-MAX_RANGE, MAX_RANGE], [0, 0], pen=pg.mkPen(color='g', width=2), name='Y轴')
    height_plot_widget.addItem(y_axis_line_height)
    # Z轴（蓝色，垂直朝上）- Y轴位置显示Z值，范围-1m到1m
    z_axis_line = pg.PlotDataItem([0, 0], [-HEIGHT_MAX_RANGE, HEIGHT_MAX_RANGE], pen=pg.mkPen(color='b', width=2), name='Z轴')
    height_plot_widget.addItem(z_axis_line)
    
    # 标注原点
    origin_text_height = pg.TextItem('雷达原点 (0,0)', anchor=(0.5, 1.5), color='blue')
    origin_text_height.setPos(0, 0)
    # 设置字体大小
    font_height = QtGui.QFont()
    font_height.setPointSize(10)
    origin_text_height.setFont(font_height)
    height_plot_widget.addItem(origin_text_height)
    
    # 标注Y轴方向（水平，左侧）
    y_axis_text_height = pg.TextItem('Y轴 (左侧)', anchor=(0.5, 0.5), color='green')
    y_axis_text_height.setPos(-MAX_RANGE * 0.7, 0.2)  # 放在左侧
    y_axis_text_height.setFont(font_height)
    height_plot_widget.addItem(y_axis_text_height)
    
    # 标注Z轴方向（垂直，上方）
    z_axis_text = pg.TextItem('Z轴 (上方)', anchor=(0.5, 0.5), color='blue')
    z_axis_text.setPos(0.2, HEIGHT_MAX_RANGE * 0.7)  # 放在上方，使用HEIGHT_MAX_RANGE
    z_axis_text.setFont(font_height)
    height_plot_widget.addItem(z_axis_text)
    
    # 高度散点图
    height_plot = pg.ScatterPlotItem(size=height_point_size.value(), pen=pg.mkPen(width=1), brush=pg.mkBrush(255, 0, 0, 200))
    height_plot_widget.addItem(height_plot)
    
    # 高度连线
    height_line_ref = pg.PlotDataItem(pen=pg.mkPen(color='blue', width=2, style=QtCore.Qt.DashLine), name='高度轨迹')
    height_plot_widget.addItem(height_line_ref)
    
    # 控制网格和坐标轴的显示
    def update_height_display():
        if height_show_grid.isChecked():
            height_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        else:
            height_plot_widget.showGrid(x=False, y=False)
        
        if height_show_axes.isChecked():
            z_axis_line.show()
            y_axis_line_height.show()
            origin_text_height.show()
            z_axis_text.show()
            y_axis_text_height.show()
        else:
            z_axis_line.hide()
            y_axis_line_height.hide()
            origin_text_height.hide()
            z_axis_text.hide()
            y_axis_text_height.hide()
    
    height_show_grid.stateChanged.connect(lambda: update_height_display())
    height_show_axes.stateChanged.connect(lambda: update_height_display())
    
    # 清除高度按钮功能
    def clear_height():
        global height_history
        height_history = []
        height_plot.setData([], [])
        height_line_ref.setData([], [])
        height_info_label.setText("高度信息: 已清除")
    
    height_clear_button.clicked.connect(clear_height)

    # ---------------------------------------------------
    # lock the aspect ratio so pixels are always square
    # view_rai.setAspectLocked(True)
    # view_rti.setAspectLocked(True)
    img_rdi = pg.ImageItem(border=None)
    img_rai = pg.ImageItem(border=None)
    # 时间-距离图
    # 绘制图像（1）：通过pyqtgraph创建图像容器img_rti，内部封装了矩阵图像的显示和着色逻辑
    img_rti = pg.ImageItem(border=None)
    img_dti = pg.ImageItem(border=None)
    img_rei = pg.ImageItem(border=None)

    # Colormap
    pgColormap = pg_get_cmap('customize')
    lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
    img_rdi.setLookupTable(lookup_table)
    img_rai.setLookupTable(lookup_table)
    img_rti.setLookupTable(lookup_table)
    img_dti.setLookupTable(lookup_table)
    img_rei.setLookupTable(lookup_table)

    view_rdi.addItem(img_rdi)
    view_rai.addItem(img_rai)
    # 绘制图像（2）：将数据容器img_rti添加到画布view_rti中
    # 至此，img_rti已经“放置”到了GUI界面的时间-距离图窗口view_rti中，但还没绘图内容（img_rti容器为空）
    # 随后，后台线程processor负责从UDP监听队列中读取原始数据，解析后塞入RTIData队列中
    view_rti.addItem(img_rti)
    view_dti.addItem(img_dti)
    view_rei.addItem(img_rei)
    

    Cliportbox.arrowClicked.connect(lambda:updatacomstatus(Cliportbox)) 
    Cliportbox.currentIndexChanged.connect(lambda:setserialport(Cliportbox, com = 'CLI'))
    Dataportbox.arrowClicked.connect(lambda:updatacomstatus(Dataportbox))
    Dataportbox.currentIndexChanged.connect(lambda:setserialport(Dataportbox, com = 'Data'))
    color_.currentIndexChanged.connect(setcolor)
    radarparameters.currentIndexChanged.connect(getradarparameters)
    # send按键 信号-槽函数
    sendcfgbtn.clicked.connect(sendconfigfunc)
    # 状态面板仅显示真实跌倒监测状态
    def _on_monitor_toggled():
        _refresh_monitor_status_preview()
    monitor_button.clicked.connect(_on_monitor_toggled)
    changepage.triggered.connect(show_sub)
    # 2022/2/24 添加小型化控件 不能正常退出了
    exitbtn.clicked.connect(app.instance().exit)
    
    # 点云配置控件信号连接（这些控件值的变化会自动在update_figure中生效）
    # 不需要额外连接，因为update_figure中会读取这些控件的值



    app.instance().exec_()


    try:
        if radar_ctrl.CLIPort:
            if radar_ctrl.CLIPort.isOpen():
                radar_ctrl.StopRadar()
    except:
        pass

if __name__ == '__main__':
    print("进程启动...")
    # Queue for access data
    BinData = Queue() # 原始数据队列

    # 时间信息

    # ZHChen use 2025-05-20 ---
    RTIData = Queue() # 时间距离图队列
    # ZHChen use 2025-05-20 ---
    DTIData = Queue() # 多普勒时间队列

    # 连续过程信息
    RDIData = Queue() # 距离多普勒队列
    RAIData = Queue() # 距离方位角队列
    REIData = Queue() # 方位角俯仰角队列
    
    # 点云数据队列
    PointCloudData = Queue() # 点云数据队列
    MLFeatureData = Queue(maxsize=256) # 模型专用对齐特征队列
    
    # 点云历史缓冲区（用于时间衰减显示）
    # 存储格式: [(timestamp_ms, pointcloud), ...]
    # 其中 pointcloud 格式: [num_points, 4] -> [range, x, y, z]
    PointCloudHistory = []  # 点云历史缓冲区
    print("创建数据队列...")

    apply_runtime_cfg(get_active_config_path())

    # config DCA1000 to receive bin data
    # dca1000_cfg是类实例化的对象
    dca1000_cfg = DCA1000Config('DCA1000Config',config_address = ('192.168.33.30', 4096),
                                                FPGA_address_cfg=('192.168.33.180', 4096))
    print("配置DCA1000地址及端口参数...")

    # 启动 PyQt5 GUI主界面
    application()

    # 关闭GUI之后发生如下：
    # 当用户关闭GUI后，关闭与dca1000的UDP连接，停止接收雷达数据
    dca1000_cfg.DCA1000_close()

    if collector is not None:
        collector.join(timeout=1)
        print("UDP监听线程同步中...")


    # 总结一下就是：
    # （1）start()启动新线程，异步执行 run() 方法
    # （2）join()阻塞当前线程，等待子线程执行完毕（常用于确保数据完整性）
    # （3）join(timeout=1)限时等待，超时后当前线程继续执行


    print("end---------程序结束---------end")
    sys.exit()
