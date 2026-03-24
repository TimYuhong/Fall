"""Interfaces for optional clip-based fall prediction models."""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class FallFeatureClip:
    """A synchronized radar feature clip for optional ML inference."""

    timestamp_start_ms: float
    timestamp_end_ms: float
    RT: Any = None
    DT: Any = None
    RDT: Any = None
    ART: Any = None
    ERT: Any = None
    pointcloud: Any = None
    height_history: List[Tuple] = field(default_factory=list)
    tracked_target_state: Dict[str, Any] = field(default_factory=dict)
    runtime_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLFeatureFrame:
    """One training-aligned RD/RA/RE frame produced by the realtime pipeline."""

    timestamp_start_ms: float
    timestamp_end_ms: float
    RDT: Any = None
    ART: Any = None
    ERT: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallPrediction:
    """Raw model prediction used by the realtime ML alarm pipeline."""

    available: bool = False
    label: str = ""
    score: float = 0.0
    probability: float = 0.0
    topk: Sequence[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.available:
            return "ML: unavailable"
        return f"ML: {self.label} (p={self.probability:.2f}, score={self.score:.3f})"


@dataclass
class PredictorLoadResult:
    success: bool
    predictor: "BaseFallPredictor"
    message: str
    model_path: str = ""
    contract: Optional["ModelRuntimeContract"] = None
    contract_validation: Optional["ContractValidationResult"] = None


@dataclass
class ModelRuntimeContract:
    """Feature/time contract that must match the active realtime cfg."""

    model_path: str = ""
    source: str = ""
    clip_frames: int = 0
    frame_periodicity_ms: float = 0.0
    feature_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    class_names: List[str] = field(default_factory=list)
    positive_labels: List[str] = field(default_factory=list)
    cfg_sha256: str = ""


@dataclass
class ContractValidationResult:
    """Outcome of validating the active runtime cfg against a model contract."""

    valid: bool = False
    status: str = "disabled"
    message: str = "ML disabled"
    mismatches: List[str] = field(default_factory=list)


class BaseFallPredictor:
    """Base interface for optional fall prediction models."""

    def predict(self, clip: FallFeatureClip) -> FallPrediction:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state when the tracked target is lost."""


class NullFallPredictor(BaseFallPredictor):
    """Default no-op predictor used until a real model is registered."""

    def predict(self, clip: FallFeatureClip) -> FallPrediction:
        return FallPrediction(available=False)

    def reset(self) -> None:
        pass


DEFAULT_MODEL_ENV_VARS = (
    "RADAR_MODEL_PATH",
    "RADAR_DEFAULT_MODEL_PATH",
)
DEFAULT_MODEL_PATTERNS = (
    "raca_v2_best.pth",
    "raca_best.pth",
)
_TORCH_PRELOAD_ATTEMPTED = False
_TORCH_PRELOAD_OK = False
_TORCH_PRELOAD_MESSAGE = ""


def _compute_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_feature_shapes(raw_shapes: Any) -> Dict[str, Tuple[int, ...]]:
    if not isinstance(raw_shapes, dict):
        return {}

    normalized: Dict[str, Tuple[int, ...]] = {}
    for key, value in raw_shapes.items():
        if value is None:
            continue
        try:
            normalized[str(key)] = tuple(int(dim) for dim in value)
        except Exception:
            continue
    return normalized


def _default_feature_shapes(
    runtime_cfg: Optional[Dict[str, Any]] = None,
    clip_frames: int = 100,
) -> Dict[str, Tuple[int, ...]]:
    if runtime_cfg:
        try:
            from runtime.aligned_features import build_training_feature_shapes

            return build_training_feature_shapes(runtime_cfg, clip_frames=clip_frames)
        except Exception:
            pass

    return {
        "RD": (int(clip_frames), 128, 64),
        "RA": (int(clip_frames), 361, 128),
        "RE": (int(clip_frames), 361, 128),
    }


def format_contract_summary(contract: Optional[ModelRuntimeContract]) -> str:
    if contract is None:
        return "no contract"

    feature_shapes = contract.feature_shapes or {}
    parts = []
    if contract.clip_frames:
        parts.append(f"{contract.clip_frames} frames")
    if contract.frame_periodicity_ms:
        parts.append(f"{contract.frame_periodicity_ms:g} ms")
    if feature_shapes.get("RD"):
        parts.append(f"RD={feature_shapes['RD']}")
    if feature_shapes.get("RA"):
        parts.append(f"RA={feature_shapes['RA']}")
    if feature_shapes.get("RE"):
        parts.append(f"RE={feature_shapes['RE']}")
    if contract.source:
        parts.append(f"source={contract.source}")
    return " | ".join(parts) if parts else "contract available"


def load_model_runtime_contract(
    model_path: str,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[ModelRuntimeContract]:
    if not model_path:
        return None

    abs_model_path = os.path.abspath(model_path)
    model_dir = os.path.dirname(abs_model_path)
    meta_path = os.path.join(model_dir, "model_meta.json")

    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        feature_shapes = _normalize_feature_shapes(meta.get("feature_shapes", {}))
        clip_frames = int(meta.get("clip_frames", 0) or 0)
        if clip_frames <= 0:
            clip_frames = 100
        if not feature_shapes:
            feature_shapes = _default_feature_shapes(runtime_cfg, clip_frames=clip_frames)

        frame_periodicity_ms = float(meta.get("frame_periodicity_ms", 0.0) or 0.0)
        if frame_periodicity_ms <= 0:
            frame_periodicity_ms = 50.0

        return ModelRuntimeContract(
            model_path=abs_model_path,
            source="model_meta.json",
            clip_frames=clip_frames,
            frame_periodicity_ms=frame_periodicity_ms,
            feature_shapes=feature_shapes,
            class_names=[str(name) for name in meta.get("class_names", [])],
            positive_labels=[str(name) for name in meta.get("positive_labels", [])],
            cfg_sha256=str(meta.get("cfg_sha256", "")).strip(),
        )

    clip_frames = 100
    return ModelRuntimeContract(
        model_path=abs_model_path,
        source="checkpoint-defaults",
        clip_frames=clip_frames,
        frame_periodicity_ms=50.0,
        feature_shapes=_default_feature_shapes(None, clip_frames=clip_frames),
    )


def validate_runtime_contract(
    contract: Optional[ModelRuntimeContract],
    runtime_cfg: Optional[Dict[str, Any]],
) -> ContractValidationResult:
    if contract is None:
        return ContractValidationResult(
            valid=False,
            status="disabled",
            message="ML disabled",
        )

    if not runtime_cfg:
        return ContractValidationResult(
            valid=False,
            status="disabled",
            message="ML disabled: runtime cfg unavailable",
        )

    mismatches: List[str] = []
    feature_shapes = contract.feature_shapes or {}
    rd_shape = feature_shapes.get("RD")
    ra_shape = feature_shapes.get("RA")
    re_shape = feature_shapes.get("RE")

    actual_range_bins = int(runtime_cfg.get("num_range_bins", 0) or 0)
    actual_doppler_bins = int(runtime_cfg.get("num_doppler_bins", 0) or 0)
    actual_frame_periodicity_ms = float(runtime_cfg.get("frame_periodicity_ms", 0.0) or 0.0)

    try:
        from runtime.aligned_features import resolve_angle_bins

        actual_angle_bins = int(resolve_angle_bins(runtime_cfg))
    except Exception:
        actual_angle_bins = 0

    if rd_shape and len(rd_shape) >= 3:
        expected_range_bins = int(rd_shape[1])
        expected_doppler_bins = int(rd_shape[2])
        if actual_range_bins and actual_range_bins != expected_range_bins:
            mismatches.append(f"range_bins={actual_range_bins} expected {expected_range_bins}")
        if actual_doppler_bins and actual_doppler_bins != expected_doppler_bins:
            mismatches.append(f"doppler_bins={actual_doppler_bins} expected {expected_doppler_bins}")

    expected_angle_bins = 0
    if ra_shape and len(ra_shape) >= 3:
        expected_angle_bins = int(ra_shape[1])
        expected_ra_range_bins = int(ra_shape[2])
        if actual_range_bins and actual_range_bins != expected_ra_range_bins:
            mismatches.append(f"RA range_bins={actual_range_bins} expected {expected_ra_range_bins}")
    if re_shape and len(re_shape) >= 3:
        expected_angle_bins = max(expected_angle_bins, int(re_shape[1]))
        expected_re_range_bins = int(re_shape[2])
        if actual_range_bins and actual_range_bins != expected_re_range_bins:
            mismatches.append(f"RE range_bins={actual_range_bins} expected {expected_re_range_bins}")
    if expected_angle_bins and actual_angle_bins and actual_angle_bins != expected_angle_bins:
        mismatches.append(f"angle_bins={actual_angle_bins} expected {expected_angle_bins}")

    if contract.frame_periodicity_ms and actual_frame_periodicity_ms:
        if abs(actual_frame_periodicity_ms - contract.frame_periodicity_ms) > 1e-6:
            mismatches.append(
                f"frame_periodicity_ms={actual_frame_periodicity_ms:g} expected {contract.frame_periodicity_ms:g}"
            )

    cfg_sha256 = contract.cfg_sha256.strip()
    if cfg_sha256:
        cfg_path = str(runtime_cfg.get("config_file", "")).strip()
        if not cfg_path or not os.path.isfile(cfg_path):
            mismatches.append("cfg file unavailable for sha256 validation")
        else:
            actual_cfg_sha256 = _compute_file_sha256(cfg_path)
            if actual_cfg_sha256 != cfg_sha256:
                mismatches.append("cfg_sha256 mismatch")

    if mismatches:
        return ContractValidationResult(
            valid=False,
            status="mismatch",
            message="ML disabled: cfg mismatch",
            mismatches=mismatches,
        )

    return ContractValidationResult(
        valid=True,
        status="ready",
        message=f"ML contract validated: {format_contract_summary(contract)}",
    )


def _attach_runtime_contract(
    result: PredictorLoadResult,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> PredictorLoadResult:
    try:
        contract = load_model_runtime_contract(result.model_path or "", runtime_cfg=runtime_cfg)
        validation = validate_runtime_contract(contract, runtime_cfg)
    except Exception as exc:
        contract = None
        validation = ContractValidationResult(
            valid=False,
            status="error",
            message=f"ML contract error: {exc}",
            mismatches=[str(exc)],
        )

    result.contract = contract
    result.contract_validation = validation
    return result


def discover_default_model_path(search_roots: Sequence[str] | None = None) -> str:
    """Discover the most suitable default model artifact for realtime inference."""

    for env_name in DEFAULT_MODEL_ENV_VARS:
        candidate = os.environ.get(env_name, "").strip()
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    if search_roots is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        search_roots = [os.path.join(project_root, "checkpoints")]

    candidates: List[str] = []
    for root in search_roots:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if not os.path.isdir(abs_root):
            continue
        for pattern in DEFAULT_MODEL_PATTERNS:
            candidates.extend(
                path
                for path in glob.glob(os.path.join(abs_root, "**", pattern), recursive=True)
                if os.path.isfile(path)
            )

    if not candidates:
        return ""

    candidates = sorted(
        {os.path.abspath(path) for path in candidates},
        key=lambda path: (os.path.getmtime(path), path),
        reverse=True,
    )
    return candidates[0]


def preload_torch_runtime(force: bool = False) -> Tuple[bool, str]:
    """Load torch early so GUI DLLs do not break later model imports on Windows."""

    global _TORCH_PRELOAD_ATTEMPTED, _TORCH_PRELOAD_OK, _TORCH_PRELOAD_MESSAGE

    if _TORCH_PRELOAD_ATTEMPTED and not force:
        return _TORCH_PRELOAD_OK, _TORCH_PRELOAD_MESSAGE

    _TORCH_PRELOAD_ATTEMPTED = True
    try:
        import torch

        _TORCH_PRELOAD_OK = True
        _TORCH_PRELOAD_MESSAGE = f"torch runtime preloaded ({torch.__version__})"
        return True, _TORCH_PRELOAD_MESSAGE
    except Exception as exc:
        guidance = ""
        if os.name == "nt":
            guidance = (
                " On Windows this often happens when torch is imported after "
                "Qt/OpenGL/matplotlib DLLs have already been loaded."
            )
        _TORCH_PRELOAD_OK = False
        _TORCH_PRELOAD_MESSAGE = f"Failed to preload torch runtime: {exc}.{guidance}".strip()
        return False, _TORCH_PRELOAD_MESSAGE


class TorchModuleFallPredictor(BaseFallPredictor):
    """Optional Torch wrapper with a simple clip-based default adapter."""

    def __init__(self, module: Any, device: str = "cpu") -> None:
        self._module = module
        self._device = device
        self._torch_available = False
        try:
            import torch  # noqa: F401

            self._torch_available = True
            module.eval()
            if hasattr(module, "to"):
                module.to(device)
        except ImportError:
            pass

    def _build_input_tensor(self, clip: FallFeatureClip) -> Any:
        """Convert a feature clip into a Torch tensor.

        The default implementation keeps the contract stable while remaining
        intentionally simple. Real models should override this method.
        """

        import numpy as np
        import torch

        if clip.RT is not None:
            arr = np.array(clip.RT, dtype=np.float32)
            return torch.from_numpy(arr).unsqueeze(0).to(self._device)
        return torch.zeros(1, 1, dtype=torch.float32).to(self._device)

    def predict(self, clip: FallFeatureClip) -> FallPrediction:
        if not self._torch_available:
            return FallPrediction(
                available=False,
                metadata={"reason": "torch not installed"},
            )

        try:
            import torch
            import torch.nn.functional as F

            input_tensor = self._build_input_tensor(clip)
            with torch.no_grad():
                output = self._module(input_tensor)

            probs = F.softmax(output, dim=-1)
            probs_np = probs.detach().cpu().numpy().reshape(-1)
            if probs_np.size == 0:
                return FallPrediction(
                    available=False,
                    metadata={"reason": "empty model output"},
                )

            labels = ["fall", "non-fall"]
            if probs_np.size == 1:
                fall_prob = float(probs_np[0])
                label = "fall" if fall_prob >= 0.5 else "non-fall"
                topk = [(label, fall_prob)]
            else:
                topk = []
                for index, prob in enumerate(probs_np):
                    label_name = labels[index] if index < len(labels) else f"class_{index}"
                    topk.append((label_name, float(prob)))
                topk.sort(key=lambda item: item[1], reverse=True)
                label = topk[0][0]
                fall_prob = next((prob for name, prob in topk if name == "fall"), float(topk[0][1]))

            return FallPrediction(
                available=True,
                label=label,
                score=float(output.max()),
                probability=fall_prob,
                topk=topk,
                metadata={"device": self._device},
            )
        except Exception as exc:
            return FallPrediction(
                available=False,
                metadata={"reason": f"inference error: {exc}"},
            )

    def reset(self) -> None:
        pass


def build_fall_predictor() -> BaseFallPredictor:
    """Return the active predictor instance.

    The default implementation intentionally keeps ML disabled until a concrete
    clip-based model is registered here.
    """

    return NullFallPredictor()


def _default_torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_predictor_from_path(
    model_path: str,
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> PredictorLoadResult:
    """Load a predictor from a future offline model artifact.

    Supported by default:
    - Python plugin files exporting `build_fall_predictor()` / `build_predictor()`
    - TorchScript files (`.pt`, `.jit`, `.ts`)
    - RACA v2 checkpoints (`.pth`) via `runtime.raca_predictor.load_raca_predictor()`
    """

    if not model_path:
        return _attach_runtime_contract(
            PredictorLoadResult(False, NullFallPredictor(), "No model selected."),
            runtime_cfg=runtime_cfg,
        )

    if not os.path.exists(model_path):
        return _attach_runtime_contract(
            PredictorLoadResult(
                False,
                NullFallPredictor(),
                f"Model file not found: {model_path}",
                model_path=model_path,
            ),
            runtime_cfg=runtime_cfg,
        )

    suffix = os.path.splitext(model_path)[1].lower()

    if suffix in {".pt", ".jit", ".ts", ".pth"}:
        preload_ok, preload_message = preload_torch_runtime()
        if not preload_ok:
            return _attach_runtime_contract(
                PredictorLoadResult(
                    False,
                    NullFallPredictor(),
                    preload_message,
                    model_path=model_path,
                ),
                runtime_cfg=runtime_cfg,
            )

    if suffix == ".py":
        try:
            spec = importlib.util.spec_from_file_location("external_fall_predictor", model_path)
            if spec is None or spec.loader is None:
                raise ImportError("Unable to create module spec.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "build_fall_predictor"):
                builder = module.build_fall_predictor
            elif hasattr(module, "build_predictor"):
                builder = module.build_predictor
            elif hasattr(module, "PREDICTOR"):
                predictor = module.PREDICTOR
            else:
                raise AttributeError("Plugin must export build_fall_predictor(), build_predictor(), or PREDICTOR.")

            if "builder" in locals():
                try:
                    signature = inspect.signature(builder)
                    if len(signature.parameters) >= 1:
                        predictor = builder(model_path)
                    else:
                        predictor = builder()
                except (TypeError, ValueError):
                    predictor = builder()

            if not isinstance(predictor, BaseFallPredictor):
                raise TypeError("Loaded plugin did not return a BaseFallPredictor instance.")

            return _attach_runtime_contract(
                PredictorLoadResult(
                    True,
                    predictor,
                    f"Loaded predictor plugin: {os.path.basename(model_path)}",
                    model_path=model_path,
                ),
                runtime_cfg=runtime_cfg,
            )
        except Exception as exc:
            return _attach_runtime_contract(
                PredictorLoadResult(
                    False,
                    NullFallPredictor(),
                    f"Failed to load predictor plugin: {exc}",
                    model_path=model_path,
                ),
                runtime_cfg=runtime_cfg,
            )

    if suffix in {".pt", ".jit", ".ts"}:
        try:
            import torch
            device = _default_torch_device()

            module = torch.jit.load(model_path, map_location=device)
            predictor = TorchModuleFallPredictor(module, device=device)
            return _attach_runtime_contract(
                PredictorLoadResult(
                    True,
                    predictor,
                    f"Loaded TorchScript model: {os.path.basename(model_path)}",
                    model_path=model_path,
                ),
                runtime_cfg=runtime_cfg,
            )
        except Exception as exc:
            return _attach_runtime_contract(
                PredictorLoadResult(
                    False,
                    NullFallPredictor(),
                    f"Failed to load TorchScript model: {exc}",
                    model_path=model_path,
                ),
                runtime_cfg=runtime_cfg,
            )

    if suffix == ".pth":
        from runtime.raca_predictor import load_raca_predictor

        return _attach_runtime_contract(
            load_raca_predictor(
                checkpoint=model_path,
                device=_default_torch_device(),
            ),
            runtime_cfg=runtime_cfg,
        )

    return _attach_runtime_contract(
        PredictorLoadResult(
            False,
            NullFallPredictor(),
            f"Unsupported model format: {suffix}",
            model_path=model_path,
        ),
        runtime_cfg=runtime_cfg,
    )
