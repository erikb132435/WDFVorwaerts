"""Hardwarefreie Regressionstests des vorbereiteten Audiocallback-Kerns."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import RealtimeAudioEngine as audio
import WDFSolverFast as fast
import WDFSolverVPres as reference


class RealtimeAudioEngineTests(unittest.TestCase):
    fs_hz = 48000

    def write_wav(self, path: Path, sample_count: int = 1000) -> np.ndarray:
        time_s = np.arange(sample_count, dtype=float) / self.fs_hz
        normalized = (
            0.45 * np.sin(2.0 * np.pi * 173.0 * time_s + 0.2)
            + 0.18 * np.sin(2.0 * np.pi * 611.0 * time_s - 0.4)
        )
        pcm = np.rint(normalized * 32767.0).astype(np.int16)
        wavfile.write(path, self.fs_hz, pcm)
        return pcm

    def test_simulated_callback_matches_fast_offline_pressure_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            pcm = self.write_wav(path, sample_count=1000)
            config = audio.AudioRealtimeConfig(
                fs_hz=self.fs_hz,
                block_size=256,
                preset_name="Spk-2",
                target_rms_voltage=1.5,
                output_mode="dry",
                dry_output_gain=0.2,
            )
            result = audio.run_simulated_wav(path, config)
            expected = fast.simulate_model(
                fast.FastBernardiniSpeakerSolver(
                    self.fs_hz,
                    reference.BERNARDINI_PRESETS["Spk-2"],
                ),
                result.input_voltage,
            )
            np.testing.assert_array_equal(result.output_pressure_pa, expected)
            normalized = pcm.astype(np.float64) / 32768.0
            np.testing.assert_array_equal(
                result.rendered_audio,
                np.asarray(0.2 * normalized, dtype=np.float32),
            )
            self.assertEqual(result.statistics.block_count, 4)
            self.assertAlmostEqual(result.input_rms_voltage, 1.5, places=12)

    def test_silent_mode_and_driver_statuses_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            self.write_wav(path, sample_count=600)
            status = audio.AudioCallbackStatus(
                output_underflow=True,
                priming_output=True,
            )
            result = audio.run_simulated_wav(
                path,
                audio.AudioRealtimeConfig(
                    fs_hz=self.fs_hz,
                    block_size=256,
                    output_mode="silent",
                ),
                status_by_block={1: status},
            )
            np.testing.assert_array_equal(result.rendered_audio, np.zeros(600, dtype=np.float32))
            summary = result.statistics.summary()
            self.assertEqual(summary["driver_status"]["output_underflows"], 1)
            self.assertEqual(summary["driver_status"]["priming_output_blocks"], 1)
            self.assertIn("p999_us", summary["callback"])
            self.assertIn("callback_deadline_episodes", summary)

    def test_mismatched_wav_samplerate_is_rejected_before_callback(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            self.write_wav(path)
            engine = audio.RealtimeAudioEngine(
                audio.AudioRealtimeConfig(fs_hz=44100, block_size=256)
            )
            with self.assertRaises(ValueError):
                engine.prepare_wav(path)

    def test_callback_requires_preparation_and_exact_driver_block(self):
        engine = audio.RealtimeAudioEngine(audio.AudioRealtimeConfig())
        with self.assertRaises(RuntimeError):
            engine.process_callback(np.empty((256, 1), dtype=np.float32))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.wav"
            self.write_wav(path)
            engine.prepare_wav(path)
            with self.assertRaises(ValueError):
                engine.process_callback(np.empty((128, 1), dtype=np.float32))
            with self.assertRaises(RuntimeError):
                engine.result()


if __name__ == "__main__":
    unittest.main()
