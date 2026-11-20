"""
File: HeteroRev round producer.

A background thread generates the next test case, its inputs, boosted inputs, and
fast-path contract traces while the main thread is waiting for the hardware executor.
The model and RNG state used by the producer are private to the thread; the main
thread keeps the original model/executor for the full confirmation chain.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .config import CONF
from . import factory

if TYPE_CHECKING:
    from .isa_spec import InstructionSet
    from .sandbox import BaseAddrTuple
    from .tc_components.test_case_code import TestCaseProgram
    from .tc_components.test_case_data import InputData
    from .traces import CTrace


@dataclass
class PreparedRound:
    test_case: 'TestCaseProgram'
    inputs: List['InputData']
    boosted_inputs: List['InputData']
    ctraces: List['CTrace']
    generation_ms: float = 0.0
    model_ms: float = 0.0
    prepared_start_s: float = 0.0
    prepared_end_s: float = 0.0
    producer_pid: int = 0
    data_state: int = 0
    input_seeds: tuple = ()
    boosted_seeds: tuple = ()


@dataclass
class ProducerFailure:
    """Terminal message for a producer that failed after it started producing."""
    kind: str
    message: str
    traceback_text: str

    def to_exception(self) -> RuntimeError:
        return RuntimeError(
            f"round producer failed: {self.kind}: {self.message}\n{self.traceback_text}")


def restore_prefetched_state(data_gen, prepared):
    """Restore confirmation RNG state and metadata lost by ndarray pickling."""
    if len(prepared.inputs) != len(prepared.input_seeds):
        raise ValueError("Missing original-input seed metadata")
    if len(prepared.boosted_inputs) != len(prepared.boosted_seeds):
        raise ValueError("Missing boosted-input seed metadata")
    data_gen._state = prepared.data_state
    data_gen._boosting_state = prepared.data_state
    for inp, seed in zip(prepared.inputs, prepared.input_seeds):
        inp.seed = seed
    for inp, seed in zip(prepared.boosted_inputs, prepared.boosted_seeds):
        inp.seed = seed


class RoundProducer(threading.Thread):
    """Generate and model-trace rounds ahead of the hardware measurement thread."""

    def __init__(self, instruction_set: 'InstructionSet', bases: 'BaseAddrTuple',
                 num_inputs: int, asm_prefix: str, max_rounds: int,
                 out_queue: 'queue.Queue[Optional[PreparedRound]]',
                 stop_event: threading.Event) -> None:
        super().__init__(name="rvzr-round-producer", daemon=True)
        self._isa = instruction_set
        self._bases = bases
        self._num_inputs = num_inputs
        self._asm_prefix = asm_prefix
        self._max_rounds = max_rounds
        self._queue = out_queue
        self._stop_event = stop_event
        self.error: Optional[BaseException] = None
        self._error_traceback = ""
        self.produced = 0
        # Private RNG/model state; never touched by the main thread
        self._codegen = factory.get_program_generator(CONF.program_generator_seed, self._isa)
        self._datagen = factory.get_data_generator(CONF.data_generator_seed)
        self._model = factory.get_model(self._bases)

    def run(self) -> None:
        try:
            for i in range(self._max_rounds):
                if self._stop_event.is_set():
                    return
                prepared = self._prepare(i)
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(prepared, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self.produced += 1
        except BaseException as e:  # propagate to the main thread
            self.error = e
            self._error_traceback = traceback.format_exc()
        finally:
            # The termination message shares the data queue with prepared rounds,
            # so the consumer cannot observe the end marker before a failure.
            termination = None
            if self.error is not None:
                termination = ProducerFailure(
                    type(self.error).__name__, str(self.error), self._error_traceback)
            while not self._stop_event.is_set():
                try:
                    self._queue.put(termination, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _prepare(self, i: int) -> PreparedRound:
        import time
        t0_s = time.perf_counter()
        t0 = time.perf_counter_ns()
        path = f"{self._asm_prefix}_tc{i}.asm"
        test_case = self._codegen.create_test_case(path)
        inputs = self._datagen.generate(self._num_inputs, n_actors=test_case.n_actors())
        t1 = time.perf_counter_ns()
        self._model.load_test_case(test_case)

        nesting = CONF.model_min_nesting
        if CONF.inputs_per_class <= 1:
            ctraces = self._model.trace_test_case(inputs, nesting)
            boosted = inputs
            fast_ctraces = ctraces
        else:
            ctraces, taints = self._model.trace_test_case_with_taints(inputs, nesting)
            boosted = self._datagen.generate_boosted(inputs, taints, CONF.inputs_per_class)
            if CONF.enable_fast_path_model:
                fast_ctraces = ctraces * CONF.inputs_per_class
            else:
                fast_ctraces = self._model.trace_test_case(boosted, nesting)
        t2 = time.perf_counter_ns()
        return PreparedRound(test_case, inputs, boosted, fast_ctraces,
                             generation_ms=(t1 - t0) / 1e6, model_ms=(t2 - t1) / 1e6,
                             prepared_start_s=t0_s,
                             prepared_end_s=time.perf_counter(),
                             producer_pid=os.getpid(),
                             data_state=self._datagen.get_state(),
                             input_seeds=tuple(x.seed for x in inputs),
                             boosted_seeds=tuple(x.seed for x in boosted))


class ProcessRoundProducer:
    """Process-based round producer (avoids GIL contention with the measurement thread)."""

    def __init__(self, instruction_set_path: str, bases: 'BaseAddrTuple', num_inputs: int,
                 asm_prefix: str, max_rounds: int,
                 out_queue: 'multiprocessing.Queue', stop_event: 'multiprocessing.Event') -> None:
        ctx = multiprocessing.get_context("fork")
        self._process = ctx.Process(
            target=_process_main,
            args=(instruction_set_path, bases, num_inputs, asm_prefix, max_rounds,
                  out_queue, stop_event),
            name="rvzr-round-producer", daemon=True)
        self.error: Optional[BaseException] = None
        self.produced = 0

    def start(self) -> None:
        self._process.start()

    def join(self, timeout: float = 0.2) -> None:
        # The producer has no shared mutable state with the parent (it only generates and
        # model-traces rounds), so a prompt terminate is safe and avoids waiting for a
        # multiprocessing queue feeder that is blocked on a full pipe.
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(0.5)

    def failure_if_exited(self) -> Optional[BaseException]:
        """Report an exit that happened without a termination message."""
        code = self._process.exitcode
        if code is None:
            return None
        return RuntimeError(
            f"round producer exited with code {code} before sending a termination message")


def _process_main(instruction_set_path, bases, num_inputs, asm_prefix, max_rounds,
                  out_queue, stop_event) -> None:
    try:
        from .isa_spec import InstructionSet
        isa = InstructionSet(instruction_set_path, CONF.instruction_categories)
        producer = RoundProducer(isa, bases, num_inputs, asm_prefix, max_rounds,
                                 out_queue, stop_event)
        producer.run()
    except BaseException as exc:
        traceback.print_exc()
        failure = ProducerFailure(type(exc).__name__, str(exc), traceback.format_exc())
        try:
            out_queue.put(failure, timeout=1)
        except Exception:
            pass
