"""Reproduzierbarer Laufzeitvergleich von Referenz- und schnellem Python-Solver."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import WDFSolverFast as fast
import WDFSolverVPres as reference


def timed_seconds(callable_, repeats: int) -> list[float]:
    durations = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        started = time.perf_counter()
        callable_()
        durations.append(time.perf_counter() - started)
        gc.enable()
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    fs = 96000.0
    params = reference.BERNARDINI_PRESETS["Spk-2"]
    defaults = reference.get_paper_two_tone_defaults("Spk-2")
    duration_s = args.samples / fs
    excitation = reference.generate_two_tone(
        fs,
        duration_s,
        defaults["freq1_hz"],
        defaults["freq2_hz"],
        defaults["vrms"] * np.sqrt(2.0),
        phase1_rad=defaults["phase1_rad"],
        phase2_rad=defaults["phase2_rad"],
    )

    def run_reference():
        model = reference.BernardiniSpeakerSolver(fs, params)
        return reference.simulate_model(model, excitation)

    def run_fast_full_block():
        model = fast.FastBernardiniSpeakerSolver(fs, params)
        return model.process_block(excitation)

    def run_fast_small_blocks():
        model = fast.FastBernardiniSpeakerSolver(fs, params)
        output = np.empty_like(excitation)
        for start in range(0, len(excitation), args.block_size):
            stop = min(start + args.block_size, len(excitation))
            model.process_block_into(excitation[start:stop], output[start:stop])
        return output

    def measure_individual_blocks():
        model = fast.FastBernardiniSpeakerSolver(fs, params)
        output_buffer = np.empty(args.block_size, dtype=np.float64)
        durations = []
        gc.collect()
        gc.disable()
        for start in range(0, len(excitation) - args.block_size + 1, args.block_size):
            input_block = excitation[start : start + args.block_size]
            started = time.perf_counter()
            model.process_block_into(input_block, output_buffer)
            durations.append(time.perf_counter() - started)
        gc.enable()
        return np.asarray(durations, dtype=np.float64)

    # Ein nicht gewerteter Vorlauf stabilisiert Import-, Cache- und Allokationseffekte.
    run_reference()
    run_fast_full_block()
    run_fast_small_blocks()

    expected = run_reference()
    fast_full = run_fast_full_block()
    fast_blocked = run_fast_small_blocks()
    reference_times = timed_seconds(run_reference, args.repeats)
    fast_full_times = timed_seconds(run_fast_full_block, args.repeats)
    fast_block_times = timed_seconds(run_fast_small_blocks, args.repeats)
    individual_block_times = measure_individual_blocks()

    def report(label: str, durations: list[float]) -> float:
        median_s = statistics.median(durations)
        microseconds_per_sample = 1e6 * median_s / len(excitation)
        print(
            f"{label}: median={median_s:.6f} s, "
            f"{microseconds_per_sample:.6f} us/sample, "
            f"runs={[round(value, 6) for value in durations]}"
        )
        return microseconds_per_sample

    print(f"samples={len(excitation)}, fs={fs:g} Hz, repeats={args.repeats}")
    reference_us = report("reference", reference_times)
    fast_full_us = report("fast_full_block", fast_full_times)
    fast_block_us = report(f"fast_blocks_{args.block_size}", fast_block_times)
    print(f"speedup_full={reference_us / fast_full_us:.3f}x")
    print(f"speedup_blocks={reference_us / fast_block_us:.3f}x")
    print(f"deadline={1e6 / fs:.6f} us/sample")
    print(f"headroom_full={1e6 / fs / fast_full_us:.3f}x")
    print(f"headroom_blocks={1e6 / fs / fast_block_us:.3f}x")
    print(f"max_abs_error_full={np.max(np.abs(fast_full - expected)):.12g}")
    print(f"max_abs_error_blocks={np.max(np.abs(fast_blocked - expected)):.12g}")
    block_deadline_s = args.block_size / fs
    print(
        "individual_blocks_us="
        f"median:{1e6 * np.median(individual_block_times):.3f},"
        f"p95:{1e6 * np.percentile(individual_block_times, 95):.3f},"
        f"p99:{1e6 * np.percentile(individual_block_times, 99):.3f},"
        f"max:{1e6 * np.max(individual_block_times):.3f}"
    )
    print(f"block_deadline_us={1e6 * block_deadline_s:.3f}")
    print(f"measured_block_deadline_misses={np.count_nonzero(individual_block_times > block_deadline_s)}")


if __name__ == "__main__":
    main()
