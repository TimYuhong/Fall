from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


ML_PRESET_CONFIGS: Dict[str, Dict[str, float]] = {
    "灵敏": {"threshold": 0.50, "required_streak": 1},
    "中等": {"threshold": 0.50, "required_streak": 2},
    "稳健": {"threshold": 0.65, "required_streak": 2},
}


@dataclass(frozen=True)
class MLAlarmState:
    active: bool = False
    status: str = "disabled"
    streak: int = 0
    required_streak: int = 2
    threshold: float = 0.5
    last_inference_id: Optional[int] = None
    last_label: str = ""
    last_probability: float = 0.0
    message: str = "ML disabled"
    reason: str = ""


def normalize_threshold(value: Any, default: float = 0.5) -> float:
    try:
        threshold = float(value)
    except Exception:
        threshold = float(default)
    return min(0.99, max(0.0, threshold))


def normalize_required_streak(value: Any, default: int = 2) -> int:
    try:
        required_streak = int(round(float(value)))
    except Exception:
        required_streak = int(default)
    return max(1, required_streak)


def get_preset_config(name: str) -> Dict[str, float]:
    preset = ML_PRESET_CONFIGS.get(str(name or "").strip())
    if preset is None:
        preset = ML_PRESET_CONFIGS["中等"]
    return {
        "threshold": float(preset["threshold"]),
        "required_streak": int(preset["required_streak"]),
    }


def build_alarm_state(
    *,
    active: bool = False,
    status: str = "disabled",
    streak: int = 0,
    required_streak: Any = 2,
    threshold: Any = 0.5,
    last_inference_id: Optional[int] = None,
    last_label: str = "",
    last_probability: Any = 0.0,
    message: str = "",
    reason: str = "",
) -> MLAlarmState:
    normalized_threshold = normalize_threshold(threshold)
    normalized_streak = normalize_required_streak(required_streak)
    try:
        probability = float(last_probability)
    except Exception:
        probability = 0.0
    return MLAlarmState(
        active=bool(active),
        status=str(status or "disabled"),
        streak=max(0, min(int(streak), normalized_streak)),
        required_streak=normalized_streak,
        threshold=normalized_threshold,
        last_inference_id=last_inference_id,
        last_label=str(last_label or ""),
        last_probability=probability,
        message=str(message or default_message_for_status(status, reason=reason)),
        reason=str(reason or ""),
    )


def build_initial_alarm_state(
    *,
    enabled: bool = True,
    model_loaded: bool = False,
    contract_valid: bool = False,
    contract_status: str = "disabled",
    required_streak: Any = 2,
    threshold: Any = 0.5,
) -> MLAlarmState:
    return update_alarm_state(
        build_alarm_state(
            status="disabled",
            required_streak=required_streak,
            threshold=threshold,
        ),
        None,
        enabled=enabled,
        model_loaded=model_loaded,
        contract_valid=contract_valid,
        contract_status=contract_status,
        tracker_ready=False,
        threshold=threshold,
        required_streak=required_streak,
    )[0]


def default_message_for_status(status: str, *, reason: str = "", details: str = "") -> str:
    normalized_status = str(status or "").strip().lower()
    normalized_reason = str(reason or "").strip().lower()
    if normalized_status == "disabled":
        if normalized_reason == "cfg mismatch":
            return "ML disabled: cfg mismatch"
        return "ML disabled"
    if normalized_status == "warming":
        return details or "ML warming"
    if normalized_status == "ready":
        if normalized_reason == "lagging":
            return "ML lagging"
        return details or "ML ready"
    if normalized_status == "tracking_required":
        return details or "ML tracking required"
    if normalized_status == "positive_pending":
        return details or "ML pending"
    if normalized_status == "alert":
        return details or "ML FALL"
    if normalized_status == "error":
        return "ML error"
    return details or "ML disabled"


def prediction_is_positive(
    prediction: Any,
    *,
    threshold: Any = 0.5,
    positive_label: str = "fall",
) -> bool:
    if prediction is None or not getattr(prediction, "available", False):
        return False
    label = str(getattr(prediction, "label", "") or "").strip().lower()
    probability = float(getattr(prediction, "probability", 0.0) or 0.0)
    return label == str(positive_label or "fall").strip().lower() and probability >= normalize_threshold(threshold)


def _extract_prediction_fields(prediction: Any) -> Tuple[Optional[int], str, float, Dict[str, Any]]:
    metadata = dict(getattr(prediction, "metadata", {}) or {})
    inference_id = metadata.get("inference_id")
    if inference_id is not None:
        try:
            inference_id = int(inference_id)
        except Exception:
            inference_id = None
    label = str(getattr(prediction, "label", "") or "")
    try:
        probability = float(getattr(prediction, "probability", 0.0) or 0.0)
    except Exception:
        probability = 0.0
    return inference_id, label, probability, metadata


def update_alarm_state(
    state: Optional[MLAlarmState],
    prediction: Any,
    *,
    enabled: bool,
    model_loaded: bool,
    contract_valid: bool,
    contract_status: str,
    tracker_ready: bool,
    threshold: Any,
    required_streak: Any,
    positive_label: str = "fall",
    lagging: bool = False,
    error_message: str = "",
) -> Tuple[MLAlarmState, bool]:
    threshold = normalize_threshold(threshold)
    required_streak = normalize_required_streak(required_streak)
    previous_state = state or build_alarm_state(
        status="disabled",
        required_streak=required_streak,
        threshold=threshold,
    )

    def make_state(**kwargs: Any) -> MLAlarmState:
        kwargs.setdefault("required_streak", required_streak)
        kwargs.setdefault("threshold", threshold)
        return build_alarm_state(**kwargs)

    if not enabled:
        return make_state(
            active=False,
            status="disabled",
            streak=0,
            last_inference_id=previous_state.last_inference_id,
            last_label=previous_state.last_label,
            last_probability=previous_state.last_probability,
            message="ML disabled",
            reason="toggle_off",
        ), False

    if not model_loaded:
        return make_state(
            active=False,
            status="disabled",
            streak=0,
            message="ML disabled",
            reason="model_unavailable",
        ), False

    if str(contract_status or "").strip().lower() == "mismatch":
        return make_state(
            active=False,
            status="disabled",
            streak=0,
            message="ML disabled: cfg mismatch",
            reason="cfg mismatch",
        ), False

    if not contract_valid:
        return make_state(
            active=False,
            status="disabled",
            streak=0,
            message="ML disabled",
            reason=str(contract_status or "disabled"),
        ), False

    if error_message:
        return make_state(
            active=False,
            status="error",
            streak=0,
            last_inference_id=previous_state.last_inference_id,
            last_label=previous_state.last_label,
            last_probability=previous_state.last_probability,
            message="ML error",
            reason=str(error_message),
        ), False

    inference_id, label, probability, metadata = _extract_prediction_fields(prediction)
    same_inference = inference_id is not None and inference_id == previous_state.last_inference_id
    if lagging:
        return make_state(
            active=False,
            status="ready",
            streak=0,
            last_inference_id=inference_id if inference_id is not None else previous_state.last_inference_id,
            last_label=label or previous_state.last_label,
            last_probability=probability if probability > 0.0 else previous_state.last_probability,
            message="ML lagging",
            reason="lagging",
        ), False

    required_frames = int(metadata.get("required_frames", 100) or 100)
    buffered_frames = int(metadata.get("buffered_frames", 0) or 0)

    if prediction is None or not getattr(prediction, "available", False):
        if buffered_frames > 0:
            return make_state(
                active=False,
                status="warming",
                streak=0,
                last_inference_id=None,
                last_label="",
                last_probability=0.0,
                message=f"ML warming {buffered_frames}/{required_frames}",
                reason="warming",
            ), False
        return make_state(
            active=False,
            status="ready",
            streak=0,
            last_inference_id=None,
            last_label="",
            last_probability=0.0,
            message="ML ready",
            reason="ready",
        ), False

    if (
        same_inference
        and threshold == previous_state.threshold
        and required_streak == previous_state.required_streak
        and previous_state.reason != "lagging"
    ):
        return make_state(
            active=previous_state.active,
            status=previous_state.status,
            streak=previous_state.streak,
            last_inference_id=previous_state.last_inference_id,
            last_label=previous_state.last_label,
            last_probability=previous_state.last_probability,
            message=previous_state.message,
            reason=previous_state.reason,
        ), False

    positive = prediction_is_positive(
        prediction,
        threshold=threshold,
        positive_label=positive_label,
    )
    normalized_label = label or "unknown"

    if same_inference and previous_state.reason == "lagging":
        if not positive:
            return make_state(
                active=False,
                status="ready",
                streak=0,
                last_inference_id=inference_id,
                last_label=normalized_label,
                last_probability=probability,
                message=f"ML: {normalized_label} p={probability:.2f}",
                reason="lagging_recovered",
            ), False
        if not tracker_ready:
            return make_state(
                active=False,
                status="tracking_required",
                streak=0,
                last_inference_id=inference_id,
                last_label=normalized_label,
                last_probability=probability,
                message=f"ML tracking required p={probability:.2f}",
                reason="tracking_required",
            ), False
        return make_state(
            active=False,
            status="ready",
            streak=0,
            last_inference_id=inference_id,
            last_label=normalized_label,
            last_probability=probability,
            message=f"ML: {normalized_label} p={probability:.2f}",
            reason="lagging_recovered",
        ), False

    if not positive:
        return make_state(
            active=False,
            status="ready",
            streak=0,
            last_inference_id=inference_id,
            last_label=normalized_label,
            last_probability=probability,
            message=f"ML: {normalized_label} p={probability:.2f}",
            reason="non_fall",
        ), False

    if not tracker_ready:
        return make_state(
            active=False,
            status="tracking_required",
            streak=0,
            last_inference_id=inference_id,
            last_label=normalized_label,
            last_probability=probability,
            message=f"ML tracking required p={probability:.2f}",
            reason="tracking_required",
        ), False

    next_streak = 1
    if same_inference and previous_state.status in {"positive_pending", "alert"}:
        next_streak = previous_state.streak
    elif previous_state.status in {"positive_pending", "alert"}:
        next_streak = previous_state.streak + 1
    next_streak = min(next_streak, required_streak)

    if next_streak >= required_streak:
        next_state = make_state(
            active=True,
            status="alert",
            streak=next_streak,
            last_inference_id=inference_id,
            last_label=normalized_label,
            last_probability=probability,
            message=f"ML FALL p={probability:.2f}",
            reason="confirmed_fall",
        )
        return next_state, not previous_state.active

    return make_state(
        active=False,
        status="positive_pending",
        streak=next_streak,
        last_inference_id=inference_id,
        last_label=normalized_label,
        last_probability=probability,
        message=f"ML pending {next_streak}/{required_streak} p={probability:.2f}",
        reason="pending_confirmation",
    ), False


def build_alert_result(state: MLAlarmState, prediction: Any) -> Dict[str, Any]:
    metadata = dict(getattr(prediction, "metadata", {}) or {})
    return {
        "source": "ml",
        "detected": bool(state.active),
        "strategy": "ml_primary",
        "message": f"ML confirmed fall p={state.last_probability:.2f} ({state.streak}/{state.required_streak})",
        "metrics": {
            "probability": float(state.last_probability),
            "streak": int(state.streak),
            "required_streak": int(state.required_streak),
            "inference_id": state.last_inference_id,
            "threshold": float(state.threshold),
            "label": state.last_label or "fall",
        },
        "metadata": metadata,
    }
