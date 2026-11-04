"""Unit tests for the SeqRev adaptive measurement controller."""
import unittest
import numpy as np

from rvzr.traces import HTrace, CTrace, CTraceEntry, RawHTraceSample
from rvzr.config import CONF
from rvzr.measurement_controller import MeasurementController, audit_decision
from rvzr.stats import FuzzingStats


def make_htrace(values, pfc0=0):
    raw = np.zeros(len(values), dtype=RawHTraceSample)
    for i, v in enumerate(values):
        raw[i]['trace'] = v
        raw[i]['pfc0'] = pfc0
    return HTrace(raw)


class FakeExecutor:
    def __init__(self, batches):
        self.batches = batches
        self.calls = 0

    def trace_test_case(self, inputs, n_reps):
        batch = self.batches[self.calls]
        self.calls += 1
        assert len(batch) == len(inputs)
        assert all(h.sample_size() == n_reps for h in batch)
        return batch


class MeasurementControllerTest(unittest.TestCase):
    def setUp(self):
        self._old = (CONF.enable_adaptive_sampling, CONF.adaptive_pfc_abs_threshold,
                     CONF.adaptive_pfc_rel_threshold, CONF.adaptive_trigger_mode,
                     CONF.adaptive_audit_rate)
        CONF.adaptive_pfc_abs_threshold = 0.5
        CONF.adaptive_pfc_rel_threshold = -1.0
        CONF.adaptive_trigger_mode = "support"
        CONF.adaptive_audit_rate = 0.0

    def tearDown(self):
        (CONF.enable_adaptive_sampling, CONF.adaptive_pfc_abs_threshold,
         CONF.adaptive_pfc_rel_threshold, CONF.adaptive_trigger_mode,
         CONF.adaptive_audit_rate) = self._old

    def test_early_stop_when_prefix_is_identical(self):
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5]), make_htrace([5, 5])]
        ex = FakeExecutor([low])
        c = MeasurementController(ex, None, n_total=10, n_initial=2)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 1)
        self.assertEqual([h.sample_size() for h in out], [2, 2])
        self.assertFalse(c.last_audit_missed)

    def test_escalates_on_trace_difference(self):
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5]), make_htrace([5, 6])]
        high = [make_htrace([5] * 8), make_htrace([6] * 8)]
        ex = FakeExecutor([low, high])
        c = MeasurementController(ex, None, n_total=10, n_initial=2)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 2)
        self.assertEqual([h.sample_size() for h in out], [10, 10])
        self.assertTrue(c._has_observed_difference(out, ct))

    def test_escalates_on_pfc_difference(self):
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5], pfc0=1.0), make_htrace([5, 5], pfc0=9.0)]
        high = [make_htrace([5] * 8), make_htrace([5] * 8)]
        ex = FakeExecutor([low, high])
        c = MeasurementController(ex, None, n_total=10, n_initial=2)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 2)
        self.assertEqual([h.sample_size() for h in out], [10, 10])

    def test_frequency_difference_escalates(self):
        CONF.adaptive_trigger_mode = "support+frequency"
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5, 6, 6, 6]), make_htrace([5, 5, 5, 6, 6])]
        high = [make_htrace([5, 6, 6, 6, 6]), make_htrace([5, 5, 5, 6, 6])]
        ex = FakeExecutor([low, high])
        c = MeasurementController(ex, None, n_total=10, n_initial=5)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 2)
        self.assertEqual([h.sample_size() for h in out], [10, 10])

    def test_same_support_same_counts_early_stops(self):
        CONF.adaptive_trigger_mode = "support+frequency"
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5, 6, 6, 6]), make_htrace([5, 5, 6, 6, 6])]
        ex = FakeExecutor([low])
        c = MeasurementController(ex, None, n_total=10, n_initial=5)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 1)
        self.assertEqual([h.sample_size() for h in out], [5, 5])

    def test_support_mode_ignores_frequency(self):
        CONF.adaptive_trigger_mode = "support"
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5, 6, 6, 6]), make_htrace([5, 5, 5, 6, 6])]
        ex = FakeExecutor([low])
        c = MeasurementController(ex, None, n_total=10, n_initial=5)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 1)
        self.assertEqual([h.sample_size() for h in out], [5, 5])

    def test_pfc_relative_threshold_escalates(self):
        CONF.adaptive_pfc_abs_threshold = -1.0
        CONF.adaptive_pfc_rel_threshold = 0.1
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5], pfc0=1.0), make_htrace([5, 5], pfc0=2.0)]
        high = [make_htrace([5] * 8), make_htrace([5] * 8)]
        ex = FakeExecutor([low, high])
        c = MeasurementController(ex, None, n_total=10, n_initial=2)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 2)
        self.assertEqual([h.sample_size() for h in out], [10, 10])

    def test_audit_decision_stateless_and_calibrated(self):
        self.assertEqual(sum(audit_decision(10, i, 0.0) for i in range(1000)), 0)
        self.assertEqual(sum(audit_decision(10, i, 1.0) for i in range(1000)), 1000)
        n10 = sum(audit_decision(10, i, 0.1) for i in range(1000))
        self.assertGreater(n10, 70)
        self.assertLess(n10, 130)
        n50 = sum(audit_decision(10, i, 0.5) for i in range(1000))
        self.assertGreater(n50, 400)
        self.assertLess(n50, 600)
        self.assertEqual(audit_decision(10, 7, 0.1), audit_decision(10, 7, 0.1))
        seq_a = [audit_decision(10, i, 0.1) for i in range(100)]
        seq_b = [audit_decision(11, i, 0.1) for i in range(100)]
        self.assertNotEqual(seq_a, seq_b)

    def test_audit_labels_missed_candidate(self):
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5]), make_htrace([5, 5])]
        high = [make_htrace([5] * 8), make_htrace([6] * 8)]
        ex = FakeExecutor([low, high])
        c = MeasurementController(ex, None, n_total=10, n_initial=2, audit_rate=1.0)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 2)
        self.assertTrue(c.last_audit_missed)
        self.assertEqual([h.sample_size() for h in out], [10, 10])

    def test_samples_saved_counts_inputs(self):
        stat = FuzzingStats()
        before = stat.adaptive_samples_saved
        reps_before = stat.adaptive_repetition_rounds_saved
        ct = [CTrace([CTraceEntry("val", 0)])] * 2
        low = [make_htrace([5, 5]), make_htrace([5, 5])]
        ex = FakeExecutor([low])
        c = MeasurementController(ex, None, n_total=10, n_initial=2)
        c.collect([object(), object()], ct)
        self.assertEqual(stat.adaptive_samples_saved - before, 16)
        self.assertEqual(stat.adaptive_repetition_rounds_saved - reps_before, 8)

    def test_no_escalation_when_different_classes_differ(self):
        ct = [CTrace([CTraceEntry("val", 0)]), CTrace([CTraceEntry("val", 1)])]
        low = [make_htrace([5, 5]), make_htrace([7, 7])]
        ex = FakeExecutor([low])
        c = MeasurementController(ex, None, n_total=10, n_initial=2)
        out = c.collect([object(), object()], ct)
        self.assertEqual(ex.calls, 1)
        self.assertEqual([h.sample_size() for h in out], [2, 2])


if __name__ == "__main__":
    unittest.main()
