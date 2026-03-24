"""Kalman-based single-target tracking for fall detection."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Optional, Tuple

import numpy as np


TrackerPosition = Tuple[float, float, float, float]
TrackerVelocity = Tuple[float, float, float]
TrackerEvent = str

_DEFAULT_DT_SECONDS = 0.05
_MIN_DT_SECONDS = 0.02
_MAX_DT_SECONDS = 0.25
_MEASUREMENT_STD_M = 0.20
_PROCESS_ACCEL_STD_MPS2 = 2.0
_INITIAL_POSITION_STD_M = 0.35
_INITIAL_VELOCITY_STD_MPS = 1.50
_DEFAULT_MAHALANOBIS_GATE = 16.0
_DEFAULT_STABLE_HITS = 2
_DEFAULT_RELOCK_GRACE_MS = 1000.0
_DEFAULT_RELOCK_DISTANCE_M = 0.6


@dataclass(frozen=True)
class TrackerState:
    locked: bool = False
    position: Optional[TrackerPosition] = None
    predicted_position: Optional[TrackerPosition] = None
    velocity: TrackerVelocity = (0.0, 0.0, 0.0)
    covariance: Optional[np.ndarray] = None
    last_update_ms: float = 0.0
    last_measurement_ms: float = 0.0
    miss_count: int = 0
    hit_streak: int = 0
    age_frames: int = 0
    status: str = "idle"
    stable_latched: bool = False
    last_stable_position: Optional[TrackerPosition] = None
    last_stable_time_ms: float = 0.0


def reset_tracker() -> TrackerState:
    return TrackerState()


def tracker_state_to_dict(state: TrackerState) -> dict:
    return {
        "locked": state.locked,
        "position": list(state.position) if state.position is not None else None,
        "predicted_position": list(state.predicted_position) if state.predicted_position is not None else None,
        "velocity": list(state.velocity),
        "covariance": state.covariance.tolist() if state.covariance is not None else None,
        "last_update_ms": state.last_update_ms,
        "last_measurement_ms": state.last_measurement_ms,
        "miss_count": state.miss_count,
        "hit_streak": state.hit_streak,
        "age_frames": state.age_frames,
        "status": state.status,
        "stable_latched": state.stable_latched,
        "last_stable_position": list(state.last_stable_position) if state.last_stable_position is not None else None,
        "last_stable_time_ms": state.last_stable_time_ms,
    }


def is_tracker_stable(state: TrackerState, *, min_hits: int = _DEFAULT_STABLE_HITS, max_miss: int = 2) -> bool:
    if not state.locked or state.position is None:
        return False
    if int(state.miss_count) > int(max_miss):
        return False
    return bool(state.stable_latched) or int(state.hit_streak) >= int(min_hits)


def format_tracker_status(
    state: TrackerState,
    *,
    max_miss: int = 2,
    stable_hits: int = _DEFAULT_STABLE_HITS,
) -> str:
    status = str(getattr(state, "status", "") or ("tracked" if state.locked else "idle"))
    if status == "predicted":
        return f"Tracker: predicted (miss {state.miss_count}/{max_miss})"
    if status == "lost":
        return "Tracker: lost"
    if not state.locked or state.position is None:
        return "Tracker: idle"
    if not is_tracker_stable(state, min_hits=stable_hits, max_miss=max_miss):
        return f"Tracker: {status} (warming {state.hit_streak}/{stable_hits})"
    if status == "relocked":
        return "Tracker: relocked"
    return "Tracker: tracked"


def update_tracker(
    state: TrackerState,
    candidates: Iterable[TrackerPosition],
    current_time_ms: float,
    gate_m: float = 1.0,
    max_miss: int = 2,
    timeout_ms: float = 2000,
    mahalanobis_gate: float = _DEFAULT_MAHALANOBIS_GATE,
    stable_hits: int = _DEFAULT_STABLE_HITS,
    relock_grace_ms: float = _DEFAULT_RELOCK_GRACE_MS,
    relock_distance_m: float = _DEFAULT_RELOCK_DISTANCE_M,
) -> Tuple[TrackerState, TrackerEvent]:
    """Update the tracker with sorted candidate target centers."""

    valid_candidates = []
    for candidate in candidates:
        normalized = _normalize_candidate(candidate)
        if normalized is not None:
            valid_candidates.append(normalized)

    if not state.locked or state.position is None or state.covariance is None:
        if not valid_candidates:
            idle_status = "lost" if state.status == "lost" else "idle"
            return TrackerState(
                locked=False,
                position=None,
                predicted_position=None,
                velocity=(0.0, 0.0, 0.0),
                covariance=None,
                last_update_ms=state.last_update_ms,
                last_measurement_ms=state.last_measurement_ms,
                miss_count=0,
                hit_streak=0,
                age_frames=0,
                status=idle_status,
                stable_latched=False,
                last_stable_position=state.last_stable_position,
                last_stable_time_ms=state.last_stable_time_ms,
            ), idle_status

        restore_stable = _should_restore_stable_on_relock(
            state,
            valid_candidates[0],
            current_time_ms,
            relock_grace_ms=relock_grace_ms,
            relock_distance_m=relock_distance_m,
        )
        event = "locked" if state.last_update_ms <= 0 else "relocked"
        return _initialize_tracker_state(
            valid_candidates[0],
            current_time_ms,
            status=event,
            stable_hits=stable_hits,
            restore_stable=restore_stable,
            last_stable_position=state.last_stable_position,
            last_stable_time_ms=state.last_stable_time_ms,
        ), event

    state_vector = _state_vector_from_state(state)
    dt_seconds = _compute_dt_seconds(state.last_update_ms, current_time_ms)
    predicted_vector, predicted_covariance = _kalman_predict(state_vector, state.covariance, dt_seconds)
    predicted_position = _vector_to_position(predicted_vector)

    matched_candidate = _select_candidate(
        valid_candidates,
        predicted_vector,
        predicted_covariance,
        gate_m=gate_m,
        mahalanobis_gate=mahalanobis_gate,
    )

    if matched_candidate is not None:
        updated_vector, updated_covariance = _kalman_update(predicted_vector, predicted_covariance, matched_candidate)
        updated_position = _vector_to_position(updated_vector)
        hit_streak = max(1, int(state.hit_streak) + 1)
        stable_latched = bool(state.stable_latched) or hit_streak >= int(stable_hits)
        last_stable_position = state.last_stable_position
        last_stable_time_ms = state.last_stable_time_ms
        if stable_latched:
            last_stable_position = updated_position
            last_stable_time_ms = float(current_time_ms)
        return TrackerState(
            locked=True,
            position=updated_position,
            predicted_position=predicted_position,
            velocity=_vector_to_velocity(updated_vector),
            covariance=updated_covariance,
            last_update_ms=float(current_time_ms),
            last_measurement_ms=float(current_time_ms),
            miss_count=0,
            hit_streak=hit_streak,
            age_frames=int(state.age_frames) + 1,
            status="tracked",
            stable_latched=stable_latched,
            last_stable_position=last_stable_position,
            last_stable_time_ms=last_stable_time_ms,
        ), "tracked"

    miss_count = int(state.miss_count) + 1
    timed_out = False
    if state.last_measurement_ms > 0:
        timed_out = (float(current_time_ms) - float(state.last_measurement_ms)) > float(timeout_ms)
    if miss_count > int(max_miss) or timed_out:
        return TrackerState(
            locked=False,
            position=None,
            predicted_position=None,
            velocity=(0.0, 0.0, 0.0),
            covariance=None,
            last_update_ms=float(current_time_ms),
            last_measurement_ms=state.last_measurement_ms,
            miss_count=0,
            hit_streak=0,
            age_frames=0,
            status="lost",
            stable_latched=False,
            last_stable_position=_effective_last_stable_position(state),
            last_stable_time_ms=_effective_last_stable_time_ms(state),
        ), "lost"

    return TrackerState(
        locked=True,
        position=predicted_position,
        predicted_position=predicted_position,
        velocity=_vector_to_velocity(predicted_vector),
        covariance=predicted_covariance,
        last_update_ms=float(current_time_ms),
        last_measurement_ms=state.last_measurement_ms,
        miss_count=miss_count,
        hit_streak=int(state.hit_streak),
        age_frames=int(state.age_frames) + 1,
        status="predicted",
        stable_latched=bool(state.stable_latched),
        last_stable_position=_effective_last_stable_position(state),
        last_stable_time_ms=_effective_last_stable_time_ms(state),
    ), "predicted"


def _normalize_candidate(candidate: Optional[TrackerPosition]) -> Optional[TrackerPosition]:
    if candidate is None or len(candidate) < 3:
        return None
    try:
        x = float(candidate[0])
        y = float(candidate[1])
        z = float(candidate[2])
    except Exception:
        return None
    radius = sqrt(x * x + y * y + z * z)
    if radius <= 0.01:
        return None
    return (x, y, z, radius)


def _effective_last_stable_position(state: TrackerState) -> Optional[TrackerPosition]:
    if state.last_stable_position is not None:
        return state.last_stable_position
    if state.stable_latched and state.position is not None:
        return state.position
    return None


def _effective_last_stable_time_ms(state: TrackerState) -> float:
    if state.last_stable_time_ms > 0:
        return float(state.last_stable_time_ms)
    if state.stable_latched and state.last_measurement_ms > 0:
        return float(state.last_measurement_ms)
    return 0.0


def _should_restore_stable_on_relock(
    state: TrackerState,
    candidate: TrackerPosition,
    current_time_ms: float,
    *,
    relock_grace_ms: float,
    relock_distance_m: float,
) -> bool:
    last_stable_position = _effective_last_stable_position(state)
    last_stable_time_ms = _effective_last_stable_time_ms(state)
    if last_stable_position is None or last_stable_time_ms <= 0:
        return False
    if (float(current_time_ms) - float(last_stable_time_ms)) > float(relock_grace_ms):
        return False
    candidate_xyz = np.array(candidate[:3], dtype=np.float64)
    stable_xyz = np.array(last_stable_position[:3], dtype=np.float64)
    return float(np.linalg.norm(candidate_xyz - stable_xyz)) <= float(relock_distance_m)


def _compute_dt_seconds(last_update_ms: float, current_time_ms: float) -> float:
    if last_update_ms <= 0:
        return _DEFAULT_DT_SECONDS
    dt_seconds = (float(current_time_ms) - float(last_update_ms)) / 1000.0
    if not np.isfinite(dt_seconds):
        return _DEFAULT_DT_SECONDS
    return min(_MAX_DT_SECONDS, max(_MIN_DT_SECONDS, dt_seconds))


def _state_vector_from_state(state: TrackerState) -> np.ndarray:
    px, py, pz, _ = state.position or (0.0, 0.0, 0.0, 0.0)
    vx, vy, vz = state.velocity
    return np.array([px, py, pz, vx, vy, vz], dtype=np.float64)


def _vector_to_position(state_vector: np.ndarray) -> TrackerPosition:
    x = float(state_vector[0])
    y = float(state_vector[1])
    z = float(state_vector[2])
    radius = sqrt(x * x + y * y + z * z)
    return (x, y, z, radius)


def _vector_to_velocity(state_vector: np.ndarray) -> TrackerVelocity:
    return (float(state_vector[3]), float(state_vector[4]), float(state_vector[5]))


def _measurement_matrix() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _measurement_covariance() -> np.ndarray:
    variance = _MEASUREMENT_STD_M ** 2
    return np.diag([variance, variance, variance]).astype(np.float64)


def _state_transition_matrix(dt_seconds: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, dt_seconds, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt_seconds, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt_seconds],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _process_covariance(dt_seconds: float) -> np.ndarray:
    q = _PROCESS_ACCEL_STD_MPS2 ** 2
    dt2 = dt_seconds * dt_seconds
    dt3 = dt2 * dt_seconds
    dt4 = dt2 * dt2
    axis_block = np.array(
        [
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2],
        ],
        dtype=np.float64,
    ) * q

    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[np.ix_([0, 3], [0, 3])] = axis_block
    covariance[np.ix_([1, 4], [1, 4])] = axis_block
    covariance[np.ix_([2, 5], [2, 5])] = axis_block
    return covariance


def _initial_covariance() -> np.ndarray:
    pos_var = _INITIAL_POSITION_STD_M ** 2
    vel_var = _INITIAL_VELOCITY_STD_MPS ** 2
    return np.diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]).astype(np.float64)


def _initialize_tracker_state(
    candidate: TrackerPosition,
    current_time_ms: float,
    *,
    status: str,
    stable_hits: int,
    restore_stable: bool,
    last_stable_position: Optional[TrackerPosition],
    last_stable_time_ms: float,
) -> TrackerState:
    x, y, z, _ = candidate
    state_vector = np.array([x, y, z, 0.0, 0.0, 0.0], dtype=np.float64)
    position = _vector_to_position(state_vector)
    stable_latched = bool(restore_stable)
    hit_streak = int(stable_hits) if stable_latched else 1
    if stable_latched:
        last_stable_position = position
        last_stable_time_ms = float(current_time_ms)
    return TrackerState(
        locked=True,
        position=position,
        predicted_position=position,
        velocity=(0.0, 0.0, 0.0),
        covariance=_initial_covariance(),
        last_update_ms=float(current_time_ms),
        last_measurement_ms=float(current_time_ms),
        miss_count=0,
        hit_streak=hit_streak,
        age_frames=1,
        status=status,
        stable_latched=stable_latched,
        last_stable_position=last_stable_position,
        last_stable_time_ms=float(last_stable_time_ms),
    )


def _kalman_predict(
    state_vector: np.ndarray,
    covariance: np.ndarray,
    dt_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    transition = _state_transition_matrix(dt_seconds)
    process_covariance = _process_covariance(dt_seconds)
    predicted_vector = transition @ state_vector
    predicted_covariance = transition @ covariance @ transition.T + process_covariance
    return predicted_vector, predicted_covariance


def _kalman_update(
    predicted_vector: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement: TrackerPosition,
) -> Tuple[np.ndarray, np.ndarray]:
    measurement_vector = np.array(measurement[:3], dtype=np.float64)
    observation = _measurement_matrix()
    measurement_covariance = _measurement_covariance()
    innovation = measurement_vector - observation @ predicted_vector
    innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_covariance
    innovation_inverse = np.linalg.pinv(innovation_covariance)
    kalman_gain = predicted_covariance @ observation.T @ innovation_inverse
    updated_vector = predicted_vector + kalman_gain @ innovation
    identity = np.eye(predicted_covariance.shape[0], dtype=np.float64)
    updated_covariance = (
        (identity - kalman_gain @ observation)
        @ predicted_covariance
        @ (identity - kalman_gain @ observation).T
        + kalman_gain @ measurement_covariance @ kalman_gain.T
    )
    return updated_vector, updated_covariance


def _select_candidate(
    candidates: Iterable[TrackerPosition],
    predicted_vector: np.ndarray,
    predicted_covariance: np.ndarray,
    *,
    gate_m: float,
    mahalanobis_gate: float,
) -> Optional[TrackerPosition]:
    if gate_m <= 0:
        return None

    observation = _measurement_matrix()
    measurement_covariance = _measurement_covariance()
    predicted_measurement = observation @ predicted_vector
    innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_covariance
    innovation_inverse = np.linalg.pinv(innovation_covariance)

    best_candidate = None
    best_signature = None
    for index, candidate in enumerate(candidates):
        innovation = np.array(candidate[:3], dtype=np.float64) - predicted_measurement
        euclidean_distance = float(np.linalg.norm(innovation))
        if euclidean_distance > float(gate_m):
            continue
        mahalanobis_distance = float(innovation.T @ innovation_inverse @ innovation)
        if not np.isfinite(mahalanobis_distance) or mahalanobis_distance > float(mahalanobis_gate):
            continue
        signature = (mahalanobis_distance, euclidean_distance, index)
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_candidate = candidate
    return best_candidate
