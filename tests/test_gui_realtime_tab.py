"""GUI-unabhaengige Tests der Konfiguration und Darstellung des Echtzeitreiters."""

from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest

from matplotlib.figure import Figure
import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import RealtimeWDFEngine as realtime
import WDFSimulatorGUI as gui


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def statistics() -> realtime.RealtimeStatistics:
    return realtime.RealtimeStatistics(
        fs_hz=96000.0,
        block_size=128,
        sample_count=384,
        block_count=3,
        signal_duration_s=0.004,
        wall_duration_s=0.004,
        input_preparation_times_s=np.asarray([20e-6, 21e-6, 22e-6]),
        kernel_times_s=np.asarray([700e-6, 720e-6, 710e-6]),
        total_block_times_s=np.asarray([720e-6, 741e-6, 732e-6]),
        release_lateness_s=np.asarray([0.0, 1e-6, 2e-6]),
        block_deadlines_s=np.full(3, 128.0 / 96000.0),
        kernel_deadline_misses=np.asarray([False, False, False]),
        total_processing_deadline_misses=np.asarray([False, False, False]),
        scheduled_deadline_misses=np.asarray([False, False, True]),
        thread_work_unit="cycles",
        input_preparation_thread_work=np.asarray([100.0, 110.0, 105.0]),
        kernel_thread_work=np.asarray([1000.0, 1010.0, 1005.0]),
        total_block_thread_work=np.asarray([1100.0, 1120.0, 1110.0]),
    )


class GuiRealtimeTabTests(unittest.TestCase):
    def test_gui_configuration_uses_physical_rms_voltage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.wav"
            input_path.write_bytes(b"RIFF")

            class Harness:
                _collect_realtime_run = gui.WDFSimulatorGUI._collect_realtime_run

            harness = Harness()
            harness.realtime_signal_path = Value(str(input_path))
            harness.realtime_preset_name = Value("Spk-2")
            harness.realtime_block_size = Value("128")
            harness.realtime_input_vrms = Value("3.0")
            harness.realtime_maximum_duration_s = Value("2.5")
            harness.realtime_paced = Value(True)
            harness.realtime_disable_gc = Value(False)
            harness.realtime_output_storage_mode = Value(
                gui.DEFAULT_REALTIME_OUTPUT_STORAGE_MODE
            )
            harness.realtime_write_monitor = Value(True)
            harness.realtime_output_root = Value(str(root / "outputs"))

            selected_path, preset_name, config, output_root = harness._collect_realtime_run()
            self.assertEqual(selected_path, input_path.resolve())
            self.assertEqual(preset_name, "Spk-2")
            self.assertEqual(config.block_size, 128)
            self.assertAlmostEqual(config.target_rms_voltage, 3.0)
            self.assertEqual(config.maximum_duration_s, 2.5)
            self.assertEqual(config.output_storage_mode, "memory")
            self.assertEqual(output_root, (root / "outputs").resolve())

    def test_summary_reports_deadline_and_all_miss_categories(self):
        text = gui.WDFSimulatorGUI._format_realtime_summary(
            statistics(),
            input_rms_voltage=3.0,
            output_directory=Path("run"),
            paced=True,
        )
        self.assertIn("Blockfrist", text)
        self.assertIn("Kernel 0", text)
        self.assertIn("Gesamtverarbeitung 0", text)
        self.assertIn("inkl. Taktjitter 1", text)
        self.assertIn("P99,9", text)
        self.assertIn("Fehlerepisoden", text)
        self.assertIn("Threadarbeit", text)
        self.assertIn("P99/Median", text)

    def test_timing_plot_renders_without_tk_window(self):
        class Canvas:
            drawn = False

            def draw_idle(self):
                self.drawn = True

        class Harness:
            _draw_realtime_statistics = gui.WDFSimulatorGUI._draw_realtime_statistics

        harness = Harness()
        harness.realtime_figure = Figure()
        harness.realtime_canvas = Canvas()
        harness._draw_realtime_statistics(statistics())
        self.assertEqual(len(harness.realtime_figure.axes), 3)
        self.assertTrue(harness.realtime_canvas.drawn)

    def test_gui_worker_forwards_progress_and_completed_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.wav"
            fs_hz = 16000
            time_s = np.arange(320, dtype=float) / fs_hz
            samples = np.rint(0.5 * np.sin(2.0 * np.pi * 200.0 * time_s) * 32767).astype(
                np.int16
            )
            wavfile.write(input_path, fs_hz, samples)
            output_directory = root / "run"

            class Harness:
                _run_realtime_worker = gui.WDFSimulatorGUI._run_realtime_worker

            harness = Harness()
            harness._realtime_messages = queue.Queue()
            harness._realtime_cancel_event = threading.Event()
            harness._run_realtime_worker(
                input_path,
                "Spk-2",
                realtime.VirtualRealtimeConfig(
                    block_size=64,
                    target_rms_voltage=1.0,
                    paced=False,
                    write_monitor_wav=False,
                ),
                output_directory,
            )

            messages = []
            while not harness._realtime_messages.empty():
                messages.append(harness._realtime_messages.get_nowait())
            self.assertIn("progress", [kind for kind, _payload in messages])
            self.assertIn("result", [kind for kind, _payload in messages])
            result_payload = next(payload for kind, payload in messages if kind == "result")
            self.assertEqual(result_payload["statistics"].sample_count, len(samples))
            self.assertTrue((output_directory / "run_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
