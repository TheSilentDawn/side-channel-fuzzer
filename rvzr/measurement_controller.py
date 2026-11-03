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
from typing import List, Sequence, TYPE_CHECKING

from .config import CONF
from .stats import FuzzingStats

if TYPE_CHECKING:
    from .executor import Executor
    from .analyser import Analyser
    from .traces import HTrace, CTrace
    from .tc_components.test_case_data import InputData

STAT = FuzzingStats()
_ROUND_COUNTER = 0
_DEBUG_LOG = os.environ.get("RVZR_ADAPTIVE_LOG", "")


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
        self._rng = random.Random(CONF.data_generator_seed ^ 0x5E9)
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

        low = self._executor.trace_test_case(list(inputs), self._n_initial)
        max_distinct = max(len(set(int(x) for x in h.get_raw_traces())) for h in low)
        multi_inputs = sum(1 for h in low
                           if len(set(int(x) for x in h.get_raw_traces())) > 1)
        trace_diff = self._has_observed_difference(low, ctraces)
        pfc_stats = self._pfc_difference_stats(low, ctraces)
        pfc_threshold = getattr(CONF, "adaptive_pfc_abs_threshold", 0.5)
        pfc_escalate = pfc_threshold >= 0 and pfc_stats["pfc_max_abs"] > pfc_threshold
        needs_more = trace_diff or pfc_escalate
        if pfc_escalate and not trace_diff:
            STAT.adaptive_pfc_escalate += 1
        audit = (not needs_more) and (self._audit_rate > 0) and \
            (self._rng.random() < self._audit_rate)

        if not needs_more and not audit:
            STAT.adaptive_early_stop += 1
            STAT.adaptive_samples_saved += self._n_total - self._n_initial
            return low

        remaining = self._n_total - self._n_initial
        high = self._executor.trace_test_case(list(inputs), remaining)
        merged = [h.merge(x) for h, x in zip(low, high)]
        full_has_diff = self._has_observed_difference(merged, ctraces)
        if needs_more:
            STAT.adaptive_escalate += 1
        if audit:
            STAT.adaptive_audit += 1
            if full_has_diff:
                STAT.adaptive_audit_missed += 1
                self.last_audit_missed = True
        if _DEBUG_LOG:
            with open(_DEBUG_LOG, "a") as f:
                f.write(json.dumps({"round": round_id, "n_inputs": len(inputs),
                                    "n_initial": self._n_initial, "max_distinct": max_distinct,
                                    "multi_inputs": multi_inputs, "needs_more": needs_more, "trace_diff": trace_diff,
                                    "pfc_escalate": pfc_escalate,
                                    "audit": audit, "full_has_diff": full_has_diff,
                                    **pfc_stats}) + "\n")
        return merged

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
    def _has_observed_difference(htraces: Sequence['HTrace'],
                                 ctraces: Sequence['CTrace']) -> bool:
        """Return True if any ctrace class contains two distinct observed HTrace sets.

        This is deliberately based on raw samples rather than a statistical test: with a
        small prefix a chi-squared test is unreliable, while an exact set difference is a
        sufficient reason to pay for the full confirmation budget.
        """
        classes: dict = {}
        for i, htrace in enumerate(htraces):
            if htrace.is_empty() or htrace.is_corrupted_or_ignored():
                return True
            key = ctraces[i]
            signature = tuple(sorted(set(int(x) for x in htrace.get_raw_traces())))
            classes.setdefault(key, set()).add(signature)
            if len(classes[key]) > 1:
                return True
        return False
