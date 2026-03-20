from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

OccGrid = List[List[int]]
ConfGrid = List[List[float]]


def _grid_copy(g: OccGrid) -> OccGrid:
    return copy.deepcopy(g)


def _hamming_occ(a: OccGrid, b: OccGrid) -> int:
    diff = 0
    for r in range(8):
        for c in range(8):
            if a[r][c] != b[r][c]:
                diff += 1
    return diff


def _changed_cells(old: OccGrid, new: OccGrid) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for r in range(8):
        for c in range(8):
            if old[r][c] != new[r][c]:
                out.append((r, c))
    return out


def _avg_conf_on_cells(confs: ConfGrid, cells: List[Tuple[int, int]]) -> float:
    if not cells:
        return 1.0

    total = 0.0
    for r, c in cells:
        total += float(confs[r][c])
    return total / len(cells)


def _mean_conf(confs: ConfGrid) -> float:
    total = 0.0
    for r in range(8):
        for c in range(8):
            total += float(confs[r][c])
    return total / 64.0


@dataclass(slots=True)
class StabilizerDecision:
    emit_occ: Optional[OccGrid]
    reason: str
    mode: str


class StateStabilizer:
    """
    Vision -> chess handoff stabilizáló.
    Nem sakk-szabályokat kezel, hanem azt dönti el,
    hogy mikor tekintünk egy occupancy állapotot elég stabilnak
    a chess logic felé történő továbbadáshoz.
    """

    def __init__(
        self,
        *,
        buffer_size: int = 7,
        min_votes_ratio: float = 0.65,
        stable_frames: int = 3,
        emit_cooldown_s: float = 0.35,
        min_mean_conf: float = 0.50,
        min_changed_conf: float = 0.55,
        max_changed_for_move: int = 6,
        hold_changed_threshold: int = 12,
        hold_low_conf_threshold: float = 0.35,
        hold_min_duration_s: float = 0.40,
        recovery_stable_frames: int = 3,
    ):
        self.buffer_size = int(buffer_size)
        self.min_votes_ratio = float(min_votes_ratio)
        self.stable_frames = int(stable_frames)
        self.emit_cooldown_s = float(emit_cooldown_s)

        self.min_mean_conf = float(min_mean_conf)
        self.min_changed_conf = float(min_changed_conf)
        self.max_changed_for_move = int(max_changed_for_move)

        self.hold_changed_threshold = int(hold_changed_threshold)
        self.hold_low_conf_threshold = float(hold_low_conf_threshold)
        self.hold_min_duration_s = float(hold_min_duration_s)
        self.recovery_stable_frames = int(recovery_stable_frames)

        self._occ_buf: Deque[OccGrid] = deque(maxlen=self.buffer_size)
        self._conf_buf: Deque[ConfGrid] = deque(maxlen=self.buffer_size)

        self._last_emitted: Optional[OccGrid] = None
        self._candidate: Optional[OccGrid] = None
        self._candidate_run = 0

        self._mode: str = "WARMUP"
        self._hold_until: float = 0.0
        self._last_emit_time: float = 0.0
        self._recovery_run = 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def last_emitted(self) -> Optional[OccGrid]:
        return self._last_emitted

    def reset(self) -> None:
        self._occ_buf.clear()
        self._conf_buf.clear()
        self._last_emitted = None
        self._candidate = None
        self._candidate_run = 0
        self._mode = "WARMUP"
        self._hold_until = 0.0
        self._last_emit_time = 0.0
        self._recovery_run = 0

    def _majority_vote_candidate(self) -> tuple[OccGrid, float]:
        n = len(self._occ_buf)
        cand: OccGrid = [[0] * 8 for _ in range(8)]
        strengths: list[float] = []

        for r in range(8):
            for c in range(8):
                counts = [0, 0, 0]
                for k in range(n):
                    v = self._occ_buf[k][r][c]
                    if v in (0, 1, 2):
                        counts[v] += 1

                best_val = max(range(3), key=lambda cls: counts[cls])
                best_cnt = counts[best_val]

                cand[r][c] = best_val
                strengths.append(best_cnt / n)

        avg_strength = sum(strengths) / len(strengths)
        return cand, avg_strength

    def update(
        self,
        occ: Optional[OccGrid],
        confs: Optional[ConfGrid],
        *,
        now_s: Optional[float] = None,
    ) -> StabilizerDecision:
        now = time.time() if now_s is None else float(now_s)

        if occ is None or confs is None:
            return StabilizerDecision(None, "no_data", self._mode)

        self._occ_buf.append(_grid_copy(occ))
        self._conf_buf.append(copy.deepcopy(confs))

        if len(self._occ_buf) < max(3, min(self.buffer_size, 3)):
            self._mode = "WARMUP"
            return StabilizerDecision(None, f"warmup({len(self._occ_buf)})", self._mode)

        mean_conf = _mean_conf(confs)

        if self._mode == "HOLD":
            if now < self._hold_until:
                return StabilizerDecision(None, "hold_active", self._mode)

            cand, vote_strength = self._majority_vote_candidate()

            if self._candidate is not None and _hamming_occ(cand, self._candidate) == 0:
                self._recovery_run += 1
            else:
                self._candidate = cand
                self._recovery_run = 1

            if mean_conf >= self.min_mean_conf and vote_strength >= self.min_votes_ratio:
                if self._recovery_run >= self.recovery_stable_frames:
                    self._mode = "STABLE"
                    self._candidate_run = 0
                    return StabilizerDecision(None, "hold_recovered", self._mode)

            return StabilizerDecision(None, "hold_recovering", self._mode)

        if mean_conf < self.hold_low_conf_threshold:
            self._mode = "HOLD"
            self._hold_until = now + self.hold_min_duration_s
            self._recovery_run = 0
            return StabilizerDecision(None, "hold_low_conf", self._mode)

        cand, vote_strength = self._majority_vote_candidate()

        if self._candidate is not None and _hamming_occ(cand, self._candidate) == 0:
            self._candidate_run += 1
        else:
            self._candidate = cand
            self._candidate_run = 1

        if self._last_emitted is None:
            if (
                vote_strength >= self.min_votes_ratio
                and self._candidate_run >= self.stable_frames
                and mean_conf >= self.min_mean_conf
            ):
                self._last_emitted = _grid_copy(cand)
                self._mode = "STABLE"
                return StabilizerDecision(_grid_copy(self._last_emitted), "emit_initial_baseline", self._mode)

            self._mode = "CANDIDATE"
            return StabilizerDecision(None, "building_initial_baseline", self._mode)

        diff_to_last = _hamming_occ(self._last_emitted, cand)
        if diff_to_last >= self.hold_changed_threshold:
            self._mode = "HOLD"
            self._hold_until = now + self.hold_min_duration_s
            self._recovery_run = 0
            return StabilizerDecision(None, f"hold_big_jump(diff={diff_to_last})", self._mode)

        if (now - self._last_emit_time) < self.emit_cooldown_s:
            self._mode = "STABLE"
            return StabilizerDecision(None, "cooldown", self._mode)

        if vote_strength < self.min_votes_ratio:
            self._mode = "CANDIDATE"
            return StabilizerDecision(None, f"weak_vote({vote_strength:.2f})", self._mode)

        if self._candidate_run < self.stable_frames:
            self._mode = "CANDIDATE"
            return StabilizerDecision(None, f"candidate_not_persistent({self._candidate_run})", self._mode)

        if mean_conf < self.min_mean_conf:
            self._mode = "CANDIDATE"
            return StabilizerDecision(None, f"mean_conf_low({mean_conf:.2f})", self._mode)

        if diff_to_last == 0:
            self._mode = "STABLE"
            return StabilizerDecision(None, "no_change", self._mode)

        changed = _changed_cells(self._last_emitted, cand)
        changed_conf = _avg_conf_on_cells(confs, changed)

        if len(changed) > self.max_changed_for_move:
            self._mode = "CANDIDATE"
            return StabilizerDecision(None, f"too_many_changed({len(changed)})", self._mode)

        if changed_conf < self.min_changed_conf:
            self._mode = "CANDIDATE"
            return StabilizerDecision(None, f"changed_conf_low({changed_conf:.2f})", self._mode)

        self._last_emitted = _grid_copy(cand)
        self._last_emit_time = now
        self._mode = "STABLE"
        return StabilizerDecision(_grid_copy(self._last_emitted), f"emit_change(changed={len(changed)})", self._mode)