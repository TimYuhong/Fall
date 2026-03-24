"""Reserved replay interfaces for future offline radar bin playback."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ReplayLoadResult:
    success: bool
    source: "BaseReplaySource"
    message: str
    replay_path: str = ""


@dataclass
class BaseReplaySource:
    replay_path: str = ""
    runtime_cfg: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset replay position."""

    def next_frame(self) -> Any:
        """Return the next replay frame when a parser is implemented."""

        return None

    def can_stream(self) -> bool:
        return False


class NullReplaySource(BaseReplaySource):
    pass


class ReservedBinReplaySource(BaseReplaySource):
    """Placeholder replay source for future .bin playback support."""

    def can_stream(self) -> bool:
        return False


def load_replay_source(replay_path: str, runtime_cfg: Dict[str, Any] | None = None) -> ReplayLoadResult:
    """Reserve a replay source for a future offline `.bin` parser."""

    if not replay_path:
        return ReplayLoadResult(False, NullReplaySource(), "No replay file selected.")

    if not os.path.exists(replay_path):
        return ReplayLoadResult(
            False,
            NullReplaySource(),
            f"Replay file not found: {replay_path}",
            replay_path=replay_path,
        )

    if os.path.splitext(replay_path)[1].lower() != ".bin":
        return ReplayLoadResult(
            False,
            NullReplaySource(),
            "Replay placeholder currently reserves .bin files only.",
            replay_path=replay_path,
        )

    source = ReservedBinReplaySource(
        replay_path=replay_path,
        runtime_cfg=dict(runtime_cfg or {}),
    )
    message = (
        f"Replay source reserved: {os.path.basename(replay_path)}. "
        "Playback parser is not wired yet."
    )
    return ReplayLoadResult(True, source, message, replay_path=replay_path)
