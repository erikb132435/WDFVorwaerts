"""Regressionstests zwischen Referenz- und parallelem schnellem WDF-Solver."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import WDFSolverFast as fast
import WDFSolverVPres as reference


FS = 96000.0


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    if denominator == 0.0:
        return float(np.linalg.norm(actual - expected))
    return float(np.linalg.norm(actual - expected) / denominator)


class FastSolverRegressionTests(unittest.TestCase):
    def assert_numerically_equivalent(self, actual, expected, label):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        self.assertEqual(actual.shape, expected.shape, label)
        maximum_absolute_error = float(np.max(np.abs(actual - expected), initial=0.0))
        self.assertLessEqual(maximum_absolute_error, 2e-12, label)
        self.assertLessEqual(relative_l2(actual, expected), 1e-12, label)

    def make_excitation(self, preset_name: str, sample_count: int = 4096) -> np.ndarray:
        rng = np.random.RandomState(7419 if preset_name == "Spk-1" else 9137)
        time_s = np.arange(sample_count, dtype=np.float64) / FS
        if preset_name == "Spk-1":
            deterministic = (
                7.0 * np.sin(2.0 * np.pi * 28.75 * time_s + 0.31)
                + 3.0 * np.sin(2.0 * np.pi * 121.78 * time_s - 0.72)
            )
            noise_rms = 0.15
        else:
            deterministic = (
                2.4 * np.sin(2.0 * np.pi * 81.4 * time_s + 0.8472)
                + 1.8 * np.sin(2.0 * np.pi * 344.79 * time_s + 2.2179)
            )
            noise_rms = 0.05
        return deterministic + noise_rms * rng.standard_normal(sample_count)

    def test_output_matches_reference_for_both_presets(self):
        for preset_name in ("Spk-1", "Spk-2"):
            with self.subTest(preset=preset_name):
                params = reference.BERNARDINI_PRESETS[preset_name]
                excitation = self.make_excitation(preset_name)
                expected = reference.simulate_model(
                    reference.BernardiniSpeakerSolver(FS, params), excitation
                )
                actual = fast.simulate_model(
                    fast.FastBernardiniSpeakerSolver(FS, params), excitation
                )
                self.assert_numerically_equivalent(actual, expected, preset_name)

    def test_internal_traces_match_reference(self):
        params = reference.BERNARDINI_PRESETS["Spk-2"]
        excitation = self.make_excitation("Spk-2", sample_count=2048)
        expected = reference.simulate_model(
            reference.BernardiniSpeakerSolver(FS, params),
            excitation,
            record_traces=True,
        )
        actual = fast.simulate_model(
            fast.FastBernardiniSpeakerSolver(FS, params),
            excitation,
            record_traces=True,
        )
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            with self.subTest(trace=key):
                self.assert_numerically_equivalent(actual[key], expected[key], key)

    def test_block_partition_does_not_change_output(self):
        params = reference.BERNARDINI_PRESETS["Spk-2"]
        excitation = self.make_excitation("Spk-2", sample_count=5003)
        expected = fast.simulate_model(
            fast.FastBernardiniSpeakerSolver(FS, params), excitation
        )

        for block_size in (1, 7, 32, 64, 127, 512, 2048):
            with self.subTest(block_size=block_size):
                model = fast.FastBernardiniSpeakerSolver(FS, params)
                blocks = []
                for start in range(0, len(excitation), block_size):
                    blocks.append(model.process_block(excitation[start : start + block_size]))
                actual = np.concatenate(blocks)
                np.testing.assert_array_equal(actual, expected)

    def test_step_interface_matches_block_interface(self):
        params = reference.BERNARDINI_PRESETS["Spk-1"]
        excitation = self.make_excitation("Spk-1", sample_count=128)
        expected = fast.simulate_model(
            fast.FastBernardiniSpeakerSolver(FS, params), excitation
        )
        model = fast.FastBernardiniSpeakerSolver(FS, params)
        actual = np.asarray([model.step(sample) for sample in excitation])
        np.testing.assert_array_equal(actual, expected)

    def test_preallocated_output_buffer_matches_regular_block(self):
        params = reference.BERNARDINI_PRESETS["Spk-2"]
        excitation = self.make_excitation("Spk-2", sample_count=1024)
        expected = fast.simulate_model(
            fast.FastBernardiniSpeakerSolver(FS, params), excitation
        )
        output_buffer = np.empty_like(excitation)
        model = fast.FastBernardiniSpeakerSolver(FS, params)
        actual = model.process_block_into(excitation, output_buffer)
        self.assertIs(actual, output_buffer)
        np.testing.assert_array_equal(actual, expected)

    def test_reset_reproduces_fresh_instance(self):
        params = reference.BERNARDINI_PRESETS["Spk-2"]
        excitation = self.make_excitation("Spk-2", sample_count=1024)
        model = fast.FastBernardiniSpeakerSolver(FS, params)
        first = model.process_block(excitation)
        model.process_block(excitation)
        model.reset()
        after_reset = model.process_block(excitation)
        np.testing.assert_array_equal(after_reset, first)

    def test_trace_key_contract_matches_reference(self):
        self.assertEqual(set(fast.TRACE_KEYS), {
            "input_voltage",
            "output_pressure",
            "displacement",
            "diaphragm_velocity",
            "volume_velocity",
            "voice_coil_current",
        })


if __name__ == "__main__":
    unittest.main()
