"""GUI-unabhaengige Tests der Solverauswahl und Vergleichsdaten."""

from pathlib import Path
import sys
import unittest

import numpy as np
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import WDFSimulatorGUI as gui
import WDFSolverVPres as reference


class GuiSolverBackendTests(unittest.TestCase):
    @staticmethod
    def stationary_comparison_result():
        return gui.run_experiment(
            {
                "experiment": "stationary_sine",
                "preset_name": "Spk-2",
                "solver_mode": gui.COMPARE_SOLVER_MODE,
                "fs": 16000.0,
                "vrms": 1.0,
                "input_level_mode": gui.STANDARD_INPUT_LEVEL_MODE,
                "duration_s": 0.04,
                "settle_s": 0.005,
                "frequency_mode": "frei wählbar",
                "freq_hz": 200.0,
                "harmonics": 5,
                "record_internal": True,
                "reference_enabled": False,
                "reference_path": "",
            }
        )

    def test_parallel_adapter_uses_one_input_and_returns_comparison_metrics(self):
        fs = 16000.0
        params = reference.BERNARDINI_PRESETS["Spk-2"]
        time_s = np.arange(1024, dtype=float) / fs
        signal = 2.0 * np.sin(2.0 * np.pi * 137.0 * time_s + 0.4)
        execution = gui._simulate(
            fs,
            params,
            signal,
            record_internal=True,
            solver_mode=gui.COMPARE_SOLVER_MODE,
        )
        self.assertEqual(execution.data["primary_solver_backend"], "fast")
        np.testing.assert_array_equal(execution.output, execution.data["fast_solver_output"])
        self.assertIn("reference_solver_displacement", execution.data)
        self.assertIn("fast_solver_displacement", execution.data)
        self.assertLess(execution.data["solver_max_abs_error_pa"], 2e-12)
        self.assertLess(execution.data["solver_relative_l2_error"], 1e-12)
        self.assertGreater(execution.data["reference_solver_time_s"], 0.0)
        self.assertGreater(execution.data["fast_solver_time_s"], 0.0)

    def test_stationary_sine_comparison_contains_visualization_data(self):
        result = self.stationary_comparison_result()
        self.assertEqual(result.data["solver_mode"], gui.COMPARE_SOLVER_MODE)
        for key in (
            "reference_solver_output",
            "fast_solver_output",
            "solver_difference",
            "reference_solver_analysis",
            "fast_solver_analysis",
            "reference_solver_displacement",
            "fast_solver_displacement",
        ):
            self.assertIn(key, result.data)
        self.assertIn("Solververgleich", gui.PLOT_OPTIONS["stationary_sine"])
        self.assertIn("Solververgleich", gui.WDFSimulatorGUI._format_result_brief(result))

    def test_solver_comparison_plot_renders_without_tk_window(self):
        result = self.stationary_comparison_result()

        class Value:
            @staticmethod
            def get():
                return "Solververgleich"

        class Canvas:
            drawn = False

            def draw_idle(self):
                self.drawn = True

        class PlotHarness:
            _decimated = staticmethod(gui.WDFSimulatorGUI._decimated)
            _redraw_result = gui.WDFSimulatorGUI._redraw_result

        harness = PlotHarness()
        harness.result = result
        harness.plot_kind = Value()
        harness.figure = Figure()
        harness.canvas = Canvas()
        harness._redraw_result()
        self.assertEqual(len(harness.figure.axes), 2)
        self.assertTrue(harness.canvas.drawn)

    def test_fast_only_mode_preserves_existing_output_contract(self):
        fs = 16000.0
        params = reference.BERNARDINI_PRESETS["Spk-1"]
        signal = reference.generate_sine(fs, 0.02, 250.0, 1.0)
        execution = gui._simulate(
            fs,
            params,
            signal,
            record_internal=False,
            solver_mode=gui.FAST_SOLVER_MODE,
        )
        self.assertEqual(execution.output.shape, signal.shape)
        self.assertEqual(execution.traces, {})
        self.assertEqual(execution.data["primary_solver_backend"], "fast")
        self.assertIn("fast_solver_time_s", execution.data)
        self.assertNotIn("reference_solver_output", execution.data)

    def test_thd_curve_uses_selected_parallel_backends(self):
        result = gui.run_experiment(
            {
                "experiment": "thd_frequency_response",
                "preset_name": "Spk-2",
                "solver_mode": gui.COMPARE_SOLVER_MODE,
                "fs": 4000.0,
                "vrms": 0.5,
                "input_level_mode": gui.STANDARD_INPUT_LEVEL_MODE,
                "f_min_hz": 100.0,
                "f_max_hz": 400.0,
                "frequency_points": 2,
                "harmonics": 4,
            }
        )
        self.assertEqual(result.data["solver_mode"], gui.COMPARE_SOLVER_MODE)
        np.testing.assert_allclose(
            result.data["fast_solver_thd_db"],
            result.data["reference_solver_thd_db"],
            rtol=0.0,
            atol=1e-9,
        )
        self.assertLess(result.data["solver_relative_l2_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
