"""
File: Noise-aware sequential hardware measurement controller.

This module implements the Idea "SeqRev": collect a small prefix of the hardware
samples, and only pay for the full sample budget when the prefix already shows a
hardware-trace difference inside a contract equivalence class.  The decision is
conservative: any difference (including noise/outliers) escalates to the full,
original measurement, so the original analyser and confirmation chain are never
bypassed.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import List, Sequence, Tuple, TYPE_CHECKING

from .config import CONF
from .stats import FuzzingStats

if TYPE_CHECKING:
    from .executor import Executor
    from .analyser import Analyser
    from .traces import HTrace, CTrace
    from .tc_components.test_case_data import InputData

STAT = FuzzingStats()
_ROUND_COUNTER = 0
_ELIGIBLE_COUNTER = 0
_DEBUG_LOG = os.environ.get("RVZR_ADAPTIVE_LOG", "")


def audit_decision(seed: int, round_id: int, rate: float) -> bool:
    """Stateless, reproducible audit decision for one eligible round.

    The old implementation created a fresh Random(seed) for every round, so every
    round saw the same first random draw.  Deriving the draw from (seed, round_id)
    preserves reproducibility while giving independent decisions across rounds.
    """
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    mixed = (int(seed) * 1000003) ^ (int(round_id) * 9176) ^ 0x5E9A11
    return random.Random(mixed).random() < rate


class MeasurementController:
    """Two-stage, noise-aware hardware sampling controller."""

    def __init__(self, executor: 'Executor', analyser: 'Analyser', n_total: int,
                 n_initial: int, audit_rate: float = 0.0) -> None:
        assert 0 < n_initial <= n_total
        self._executor = executor
        self._analyser = analyser
        self._n_total = n_total
        self._n_initial = n_initial
        self._audit_rate = audit_rate
        self.last_audit_missed = False

    def collect(self, inputs: Sequence['InputData'], ctraces: Sequence['CTrace']) -> List['HTrace']:
        """Collect hardware traces, stopping after the prefix if it is indistinguishable."""
        if len(inputs) != len(ctraces):
            raise ValueError("inputs and ctraces must have the same length")
        global _ROUND_COUNTER
        round_id = _ROUND_COUNTER
        _ROUND_COUNTER += 1
        if self._n_total <= self._n_initial:
            return self._executor.trace_test_case(list(inputs), self._n_total)

        staging = getattr(CONF, "adaptive_staging", "fixed")
        if staging == "staged":
            return self._collect_staged(inputs, ctraces, round_id)

        low = self._executor.trace_test_case(list(inputs), self._n_initial)
        max_distinct = max(len(set(int(x) for x in h.get_raw_traces())) for h in low)
        multi_inputs = sum(1 for h in low
                           if len(set(int(x) for x in h.get_raw_traces())) > 1)
        mode = getattr(CONF, "adaptive_trigger_mode", "support")
        support_diff, freq_diff = self._difference_flags(low, ctraces)
        trace_diff = support_diff if mode == "support" else (support_diff or freq_diff)
        pfc_stats = self._pfc_difference_stats(low, ctraces)
        pfc_abs = getattr(CONF, "adaptive_pfc_abs_threshold", 0.5)
        pfc_rel = getattr(CONF, "adaptive_pfc_rel_threshold", -1.0)
        pfc_escalate = ((pfc_abs >= 0 and pfc_stats["pfc_max_abs"] > pfc_abs)
                        or (pfc_rel >= 0 and pfc_stats["pfc_max_rel"] > pfc_rel))
        needs_more = trace_diff or pfc_escalate
        if pfc_escalate and not trace_diff:
            STAT.adaptive_pfc_escalate += 1
        audit = False
        if not needs_more:
            global _ELIGIBLE_COUNTER
            audit_index = _ELIGIBLE_COUNTER
            _ELIGIBLE_COUNTER += 1
            STAT.adaptive_audit_eligible += 1
            audit = audit_decision(CONF.data_generator_seed, audit_index, self._audit_rate)

        if not needs_more and not audit:
            STAT.adaptive_early_stop += 1
            STAT.adaptive_samples_saved += (self._n_total - self._n_initial) * len(inputs)
            STAT.adaptive_repetition_rounds_saved += self._n_total - self._n_initial
            return low

        remaining = self._n_total - self._n_initial
        high = self._executor.trace_test_case(list(inputs), remaining)
        merged = [h.merge(x) for h, x in zip(low, high)]
        full_support_diff, full_freq_diff = self._difference_flags(merged, ctraces)
        full_has_diff = (full_support_diff if mode == "support"
                         else (full_support_diff or full_freq_diff))
        if needs_more:
            STAT.adaptive_escalate += 1
        if audit:
            STAT.adaptive_audit += 1
            STAT.adaptive_audit_checked += 1
            if full_has_diff:
                STAT.adaptive_audit_missed += 1
                self.last_audit_missed = True
        audit_index = locals().get("audit_index", -1)
        if _DEBUG_LOG:
            with open(_DEBUG_LOG, "a") as f:
                f.write(json.dumps({"round": round_id, "audit_index": audit_index,
                                    "n_inputs": len(inputs),
                                    "n_initial": self._n_initial, "max_distinct": max_distinct,
                                    "multi_inputs": multi_inputs, "needs_more": needs_more,
                                    "trace_diff": trace_diff, "support_diff": support_diff,
                                    "freq_diff": freq_diff, "pfc_escalate": pfc_escalate,
                                    "audit": audit, "full_has_diff": full_has_diff,
                                    "full_support_diff": full_support_diff,
                                    "full_freq_diff": full_freq_diff,
                                    **pfc_stats}) + "\n")
        return merged

    def _collect_staged(self, inputs: Sequence['InputData'],
                        ctraces: Sequence['CTrace'], round_id: int) -> List['HTrace']:
        """Progressive sampling: walk intermediate boundaries while clean, jump to
        the full budget only when a trigger fires.

        Boundaries are [n_initial] + adaptive_stage_sizes (e.g. 5 -> 7).  A round that
        stays clean through every boundary keeps only those samples (7), which is more
        than the fixed controller's 5 and therefore catches leaks that surface by the
        last boundary.  Any trigger (support/frequency/PFC) escalates straight to
        ``n_total`` because a candidate needs the full confirmation budget.
        """
        boundaries = sorted({int(x) for x in [self._n_initial]
                             + list(CONF.adaptive_stage_sizes)
                             if 0 < int(x) < self._n_total})
        mode = getattr(CONF, "adaptive_trigger_mode", "support")
        pfc_abs = getattr(CONF, "adaptive_pfc_abs_threshold", 0.5)
        pfc_rel = getattr(CONF, "adaptive_pfc_rel_threshold", -1.0)

        def _trigger(htraces):
            support_diff, freq_diff = self._difference_flags(htraces, ctraces)
            trace_diff = support_diff if mode == "support" else (support_diff or freq_diff)
            pfc_stats = self._pfc_difference_stats(htraces, ctraces)
            pfc_escalate = ((pfc_abs >= 0 and pfc_stats["pfc_max_abs"] > pfc_abs)
                            or (pfc_rel >= 0 and pfc_stats["pfc_max_rel"] > pfc_rel))
            return trace_diff or pfc_escalate, trace_diff, pfc_stats

        samples = self._executor.trace_test_case(list(inputs), boundaries[0])
        n_have = boundaries[0]
        escalated = False
        for stage_id, target in enumerate(boundaries[1:], start=1):
            fired, _, _ = _trigger(samples)
            if fired:
                break
            extra = self._executor.trace_test_case(list(inputs), target - n_have)
            samples = [h.merge(x) for h, x in zip(samples, extra)]
            n_have = target
            STAT.adaptive_stage_entries[stage_id] = \
                STAT.adaptive_stage_entries.get(stage_id, 0) + 1

        fired, trace_diff, pfc_stats = _trigger(samples)
        if fired:
            escalated = True
            if n_have < self._n_total:
                extra = self._executor.trace_test_case(list(inputs), self._n_total - n_have)
                samples = [h.merge(x) for h, x in zip(samples, extra)]
                n_have = self._n_total

        if escalated:
            STAT.adaptive_escalate += 1
        else:
            STAT.adaptive_early_stop += 1
            STAT.adaptive_samples_saved += (self._n_total - n_have) * len(inputs)
            STAT.adaptive_repetition_rounds_saved += self._n_total - n_have
        if _DEBUG_LOG:
            with open(_DEBUG_LOG, "a") as f:
                f.write(json.dumps({
                    "round": round_id, "staged": True, "boundaries": boundaries,
                    "n_samples": n_have, "escalated": escalated,
                    "final_trace_diff": trace_diff, "final_pfc": pfc_stats,
                    "n_inputs": len(inputs)}) + "\n")
        return samples

    @staticmethod
    def _pfc_difference_stats(htraces: Sequence['HTrace'], ctraces: Sequence['CTrace']) -> dict:
        """Summarize cross-input PFC differences within contract classes (early-warning signal)."""
        classes: dict = {}
        for i, htrace in enumerate(htraces):
            arr = htrace.get_raw_readings()
            means = []
            for p in ("pfc0", "pfc1", "pfc2", "pfc3", "pfc4"):
                vals = arr[p].astype(float)
                means.append(float(vals.mean()))
            classes.setdefault(ctraces[i], []).append(means)
        max_abs = 0.0
        max_rel = 0.0
        for members in classes.values():
            if len(members) < 2:
                continue
            for j in range(5):
                vals = [m[j] for m in members]
                abs_diff = max(vals) - min(vals)
                rel_diff = abs_diff / (1.0 + (abs(max(vals)) + abs(min(vals))) / 2.0)
                max_abs = max(max_abs, abs_diff)
                max_rel = max(max_rel, rel_diff)
        return {"pfc_max_abs": max_abs, "pfc_max_rel": max_rel}

    @staticmethod
    def _difference_flags(htraces: Sequence['HTrace'],
                          ctraces: Sequence['CTrace']) -> Tuple[bool, bool]:
        """Return (support_diff, frequency_diff) for a contract class.

        support_diff is the original, distribution-insensitive criterion: the set of
        observed trace values differs between inputs.  frequency_diff also compares the
        per-input value counts, which is the information a chi-squared analyser uses.
        With a small prefix this is intentionally conservative: any count difference
        escalates to the full budget rather than claiming equivalence.
        """
        classes: dict = {}
        for i, htrace in enumerate(htraces):
            if htrace.is_empty() or htrace.is_corrupted_or_ignored():
                return True, True
            raw = tuple(int(x) for x in htrace.get_raw_traces())
            classes.setdefault(ctraces[i], []).append((frozenset(raw), Counter(raw)))
        support_diff = False
        freq_diff = False
        for members in classes.values():
            if len(members) < 2:
                continue
            support0, counts0 = members[0]
            for support, counts in members[1:]:
                if support != support0:
                    support_diff = True
                if counts != counts0:
                    freq_diff = True
        return support_diff, freq_diff

    @classmethod
    def _has_observed_difference(cls, htraces: Sequence['HTrace'],
                                 ctraces: Sequence['CTrace']) -> bool:
        """Backward-compatible wrapper used by tests/diagnostics (support mode)."""
        support_diff, _ = cls._difference_flags(htraces, ctraces)
        return support_diff
