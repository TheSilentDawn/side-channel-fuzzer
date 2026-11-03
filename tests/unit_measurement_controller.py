"""Unit tests for the SeqRev adaptive measurement controller."""
import unittest
import numpy as np

from rvzr.traces import HTrace, CTrace, CTraceEntry, RawHTraceSample
from rvzr.config import CONF
from rvzr.measurement_controller import MeasurementController


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
        self._old = (CONF.enable_adaptive_sampling, CONF.adaptive_pfc_abs_threshold)
        CONF.adaptive_pfc_abs_threshold = 0.5

    def tearDown(self):
        CONF.enable_adaptive_sampling, CONF.adaptive_pfc_abs_threshold = self._old

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
