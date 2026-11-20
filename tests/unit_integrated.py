"""Correctness checks for combined sampling, metadata transfer and confirmation."""
import multiprocessing
import os
import pickle
import queue
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from rvzr.config import CONF
from rvzr.data_generator import DataGenerator
from rvzr.fuzzer import _RoundManager, _get_prepared_round
from rvzr import pipeline as pipeline_module
from rvzr.pipeline import PreparedRound, ProducerFailure, restore_prefetched_state
from rvzr.tc_components.test_case_data import InputTaint
from rvzr.traces import HTrace, RawHTraceSample


class IntegratedTests(unittest.TestCase):
    def test_confirmation_reboost_reuses_producer_rng(self):
        producer = DataGenerator(12345)
        inputs = producer.generate(2, 1)
        taints = [InputTaint(1), InputTaint(1)]
        boosted = producer.generate_boosted(inputs, taints, 2)
        prepared = PreparedRound(None, inputs, boosted, [], data_state=producer.get_state(),
                                 input_seeds=tuple(x.seed for x in inputs),
                                 boosted_seeds=tuple(x.seed for x in boosted))
        received = pickle.loads(pickle.dumps(prepared))
        consumer = DataGenerator(12345)
        wrong = consumer.generate_boosted(received.inputs, taints, 2)
        self.assertNotEqual(wrong[-1].tobytes(), boosted[-1].tobytes())
        restore_prefetched_state(consumer, received)
        correct = consumer.generate_boosted(received.inputs, taints, 2)
        self.assertEqual([x.tobytes() for x in correct], [x.tobytes() for x in boosted])
        self.assertEqual([x.seed for x in received.inputs], list(prepared.input_seeds))
        self.assertEqual([x.seed for x in received.boosted_inputs], list(prepared.boosted_seeds))

    def test_missing_seed_metadata_rejected(self):
        prepared = PreparedRound(None, [object()], [], [], data_state=1)
        with self.assertRaises(ValueError):
            restore_prefetched_state(DataGenerator(1), prepared)

    def test_preparing_next_round_does_not_change_snapshot(self):
        dg = DataGenerator(54321)
        first = dg.generate(2, 1)
        prepared = PreparedRound(None, first, first, [], data_state=dg.get_state(),
                                 input_seeds=tuple(x.seed for x in first),
                                 boosted_seeds=tuple(x.seed for x in first))
        dg.generate(8, 1)
        consumer = DataGenerator(0)
        restore_prefetched_state(consumer, prepared)
        self.assertEqual(consumer.get_state(), 54323)

    def split_manager(self, initial=True, extend=False):
        calls = []
        def trace(inputs, n):
            calls.append(n)
            raw = np.zeros(n, dtype=RawHTraceSample)
            raw['trace'] = len(calls)
            return [HTrace(raw)]
        manager = object.__new__(_RoundManager)
        manager.fuzzer = SimpleNamespace(executor=SimpleNamespace(trace_test_case=trace))
        manager.conf = SimpleNamespace(is_initial=initial, extend_htraces=extend, executor_n_reps=10)
        manager.boosted_inputs = [object()]
        manager.htraces = [HTrace(np.zeros(2, dtype=RawHTraceSample))]
        return manager, calls

    def test_forced_split_preserves_all_samples(self):
        old = CONF.enable_forced_split
        CONF.enable_forced_split = True
        try:
            manager, calls = self.split_manager()
            manager._collect_htraces()
            self.assertEqual(calls, [5, 5])
            self.assertEqual(sorted(manager.htraces[0].get_raw_traces()), [1]*5 + [2]*5)
        finally:
            CONF.enable_forced_split = old

    def test_forced_split_does_not_replace_later_confirmation(self):
        old = CONF.enable_forced_split
        CONF.enable_forced_split = True
        try:
            manager, calls = self.split_manager(initial=False)
            manager._collect_htraces()
            self.assertEqual(calls, [10])
        finally:
            CONF.enable_forced_split = old

    def test_forced_split_does_not_replace_noise_extension(self):
        old = CONF.enable_forced_split
        CONF.enable_forced_split = True
        try:
            manager, calls = self.split_manager(extend=True)
            manager._collect_htraces()
            self.assertEqual(calls, [10])
            self.assertEqual(manager.htraces[0].sample_size(), 12)
        finally:
            CONF.enable_forced_split = old

    def _run_isolated_producer(self, producer_cls, max_rounds, expected_items):
        original = pipeline_module.RoundProducer
        pipeline_module.RoundProducer = producer_cls
        ctx = multiprocessing.get_context("fork")
        out_queue = ctx.Queue(maxsize=4)
        stop_event = ctx.Event()
        producer = pipeline_module.ProcessRoundProducer(
            str(Path(__file__).parent / "x86_tests" / "min_x86.json"),
            (0, 0, 0, 0), 1, "/tmp/unused-prefix", max_rounds, out_queue, stop_event)
        try:
            producer.start()
            items = [out_queue.get(timeout=5) for _ in range(expected_items)]
            producer.join(timeout=5)
            return items, producer
        finally:
            pipeline_module.RoundProducer = original

    @staticmethod
    def _isolated_producer_class(prepare):
        class IsolatedProducer(pipeline_module.RoundProducer):
            def __init__(self, isa, bases, num_inputs, asm_prefix, max_rounds,
                         out_queue, stop_event):
                # Keep the real RoundProducer.run(); only bypass model setup.
                self._max_rounds = max_rounds
                self._queue = out_queue
                self._stop_event = stop_event
                self.error = None
                self._error_traceback = ""
                self.produced = 0

            def _prepare(self, i):
                return prepare(i)

        return IsolatedProducer

    def test_process_producer_propagates_prepare_exception(self):
        def prepare(_):
            raise IndexError("intentional _prepare failure")

        producer_cls = self._isolated_producer_class(prepare)
        items, producer = self._run_isolated_producer(producer_cls, 1, 1)
        failure = items[0]
        self.assertIsInstance(failure, ProducerFailure)
        self.assertEqual(failure.kind, "IndexError")
        self.assertIn("intentional _prepare failure", failure.message)
        self.assertIn("Traceback", failure.traceback_text)
        # The child reported the error and exited normally; the parent still has it.
        self.assertEqual(producer._process.exitcode, 0)

    def test_process_producer_failure_ordered_after_rounds(self):
        def prepare(i):
            if i == 0:
                return PreparedRound(None, [], [], [])
            raise IndexError("intentional failure after one round")

        producer_cls = self._isolated_producer_class(prepare)
        items, _ = self._run_isolated_producer(producer_cls, 2, 2)
        self.assertIsInstance(items[0], PreparedRound)
        self.assertIsInstance(items[1], ProducerFailure)
        self.assertEqual(items[1].kind, "IndexError")

    def test_wait_for_producer_reads_exit_code(self):
        class EmptyQueue:
            def get(self, timeout):
                raise queue.Empty

            def get_nowait(self):
                raise queue.Empty

        class ExitedProducer:
            error = None

            def failure_if_exited(self):
                return RuntimeError(
                    "round producer exited with code 7 before sending a termination message")

        with self.assertRaises(RuntimeError) as raised:
            _get_prepared_round(EmptyQueue(), ExitedProducer(), timeout=0.01)
        self.assertIn("code 7", str(raised.exception))

    def test_wait_prefers_queued_failure_over_exit_code(self):
        failure = ProducerFailure("ValueError", "original error", "traceback")

        class EmptyThenMessageQueue:
            def get(self, timeout):
                raise queue.Empty

            def get_nowait(self):
                return failure

        class ExitedProducer:
            error = None

            def failure_if_exited(self):
                return RuntimeError(
                    "round producer exited with code 0 before sending a termination message")

        message = _get_prepared_round(EmptyThenMessageQueue(), ExitedProducer(), timeout=0.01)
        self.assertIs(message, failure)

    def test_process_producer_reports_exit_without_message(self):
        def prepare(_):
            os._exit(7)

        producer_cls = self._isolated_producer_class(prepare)
        original = pipeline_module.RoundProducer
        pipeline_module.RoundProducer = producer_cls
        ctx = multiprocessing.get_context("fork")
        out_queue = ctx.Queue(maxsize=2)
        stop_event = ctx.Event()
        producer = pipeline_module.ProcessRoundProducer(
            str(Path(__file__).parent / "x86_tests" / "min_x86.json"),
            (0, 0, 0, 0), 1, "/tmp/unused-prefix", 1, out_queue, stop_event)
        try:
            producer.start()
            producer.join(timeout=5)
            failure = producer.failure_if_exited()
            self.assertIsNotNone(failure)
            self.assertIn("code 7", str(failure))
            self.assertEqual(producer._process.exitcode, 7)
        finally:
            pipeline_module.RoundProducer = original


if __name__ == '__main__':
    unittest.main()
