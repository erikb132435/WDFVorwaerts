"""Regressionstests des virtuellen blockweisen Echtzeitpfads."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import RealtimeWDFEngine as realtime
import WDFSolverFast as fast
import WDFSolverVPres as reference


class RealtimeWDFEngineTests(unittest.TestCase):
    fs_hz = 16000

    def write_test_wav(self, path: Path, duration_s: float = 0.05) -> np.ndarray:
        time_s = np.arange(int(self.fs_hz * duration_s), dtype=float) / self.fs_hz
        waveform = (
            0.55 * np.sin(2.0 * np.pi * 137.0 * time_s + 0.2)
            + 0.2 * np.sin(2.0 * np.pi * 419.0 * time_s - 0.4)
        )
        pcm = np.rint(np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(path, self.fs_hz, pcm)
        return pcm

    def test_block_stream_matches_fast_offline_solver_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            pcm = self.write_test_wav(path)
            config = realtime.VirtualRealtimeConfig(
                block_size=64,
                target_rms_voltage=1.5,
                paced=False,
                write_monitor_wav=False,
            )
            result = realtime.run_virtual_realtime_wav(path, "Spk-2", config=config)

            normalized = pcm.astype(np.float64) / 32768.0
            voltage = normalized * result.voltage_gain
            expected = fast.simulate_model(
                fast.FastBernardiniSpeakerSolver(
                    self.fs_hz, reference.BERNARDINI_PRESETS["Spk-2"]
                ),
                voltage,
            )
            np.testing.assert_array_equal(result.output_pressure_pa, expected)
            self.assertAlmostEqual(result.input_rms_voltage, 1.5, places=12)
            self.assertEqual(result.statistics.sample_count, len(pcm))
            self.assertEqual(result.statistics.block_count, 13)
            self.assertTrue(np.all(np.isfinite(result.statistics.kernel_times_s)))
            self.assertTrue(
                np.all(np.isfinite(result.statistics.kernel_thread_work))
            )
            self.assertTrue(
                np.all(result.statistics.kernel_thread_work >= 0.0)
            )
            self.assertIn(result.statistics.thread_work_unit, {"cycles", "ns"})
            self.assertGreater(float(np.median(result.statistics.kernel_thread_work)), 0.0)

    def test_artifact_directory_contains_scientific_output_monitor_and_timings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.wav"
            pcm = self.write_test_wav(input_path, duration_s=0.03)
            output_directory = root / "run"
            config = realtime.VirtualRealtimeConfig(
                block_size=128,
                target_rms_voltage=1.0,
                paced=False,
                write_monitor_wav=True,
            )
            result = realtime.run_virtual_realtime_wav(
                input_path,
                "Spk-1",
                config=config,
                output_directory=output_directory,
            )
            try:
                for filename in (
                    "output_pressure_pa.npy",
                    "monitor_normalized.wav",
                    "block_timing.csv",
                    "run_metadata.json",
                ):
                    self.assertTrue((output_directory / filename).is_file(), filename)

                stored_pressure = np.load(output_directory / "output_pressure_pa.npy")
                np.testing.assert_array_equal(stored_pressure, result.output_pressure_pa)
                monitor_fs, monitor = wavfile.read(output_directory / "monitor_normalized.wav")
                self.assertEqual(monitor_fs, self.fs_hz)
                self.assertEqual(len(monitor), len(pcm))
                self.assertEqual(monitor.dtype, np.int16)

                metadata = json.loads(
                    (output_directory / "run_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["outputs"]["scientific_pressure"]["unit"], "Pa")
                self.assertEqual(
                    metadata["outputs"]["scientific_pressure"]["stream_buffer"],
                    "memmap",
                )
                self.assertFalse(metadata["outputs"]["monitor_wav"]["scientific_level_valid"])
                self.assertEqual(metadata["statistics"]["sample_count"], len(pcm))
                self.assertIn("kernel_thread_work", metadata["statistics"])
                timing_lines = (output_directory / "block_timing.csv").read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertEqual(len(timing_lines), result.statistics.block_count + 1)
                self.assertIn("kernel_thread_work_", timing_lines[0])
            finally:
                result.close()

    def test_memory_output_is_saved_only_after_stream_and_preserves_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.wav"
            self.write_test_wav(input_path, duration_s=0.03)
            output_directory = root / "run"
            result = realtime.run_virtual_realtime_wav(
                input_path,
                "Spk-2",
                config=realtime.VirtualRealtimeConfig(
                    block_size=64,
                    target_rms_voltage=1.0,
                    paced=False,
                    write_monitor_wav=False,
                    output_storage_mode="memory",
                ),
                output_directory=output_directory,
            )
            try:
                self.assertNotIsInstance(result.output_pressure_pa, np.memmap)
                self.assertEqual(result.output_storage_mode, "memory")
                self.assertTrue((output_directory / "output_pressure_pa.npy").is_file())
                stored = np.load(output_directory / "output_pressure_pa.npy")
                np.testing.assert_array_equal(stored, result.output_pressure_pa)
                metadata = json.loads(
                    (output_directory / "run_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["config"]["output_storage_mode"], "memory")
                self.assertEqual(
                    metadata["outputs"]["scientific_pressure"]["stream_buffer"],
                    "memory",
                )
            finally:
                result.close()

    def test_invalid_output_storage_mode_is_rejected(self):
        config = realtime.VirtualRealtimeConfig(output_storage_mode="disk")
        with self.assertRaises(ValueError):
            config.validate()

    def test_paced_run_spans_at_least_the_signal_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            self.write_test_wav(path, duration_s=0.025)
            config = realtime.VirtualRealtimeConfig(
                block_size=128,
                target_rms_voltage=0.5,
                paced=True,
                write_monitor_wav=False,
            )
            result = realtime.run_virtual_realtime_wav(path, "Spk-2", config=config)
            self.assertGreaterEqual(
                result.statistics.wall_duration_s,
                0.98 * result.statistics.signal_duration_s,
            )
            self.assertEqual(
                result.statistics.scheduled_deadline_misses.shape,
                (result.statistics.block_count,),
            )

    def test_progress_callback_reports_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            pcm = self.write_test_wav(path, duration_s=0.03)
            progress = []
            result = realtime.run_virtual_realtime_wav(
                path,
                "Spk-2",
                config=realtime.VirtualRealtimeConfig(
                    block_size=64,
                    paced=False,
                    write_monitor_wav=False,
                ),
                progress_callback=progress.append,
            )
            self.assertGreaterEqual(len(progress), 1)
            self.assertEqual(progress[-1].completed_blocks, result.statistics.block_count)
            self.assertEqual(progress[-1].completed_samples, len(pcm))

    def test_pre_requested_cancellation_removes_incomplete_pressure_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.wav"
            self.write_test_wav(path, duration_s=0.03)
            output_directory = root / "cancelled"
            cancellation = threading.Event()
            cancellation.set()
            with self.assertRaises(realtime.RealtimeRunCancelled):
                realtime.run_virtual_realtime_wav(
                    path,
                    "Spk-1",
                    config=realtime.VirtualRealtimeConfig(
                        block_size=64,
                        paced=False,
                        write_monitor_wav=False,
                    ),
                    output_directory=output_directory,
                    cancel_event=cancellation,
                )
            self.assertFalse((output_directory / "output_pressure_pa.npy").exists())


if __name__ == "__main__":
    unittest.main()
