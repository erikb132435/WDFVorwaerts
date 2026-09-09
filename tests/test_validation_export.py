"""Tests fuer reproduzierbare wissenschaftliche Validierungspakete."""

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import spectral_comparison as comparison
import validation_export


class ValidationExportTests(unittest.TestCase):
    def make_time_reference(self, path: Path) -> comparison.ReferenceData:
        path.write_text("1\n2\n3\n", encoding="utf-8")
        return comparison.ReferenceData(
            path=path,
            figure="7b",
            preset_name="Spk-2",
            speaker_model="FSN020.71-4",
            source_type="measurement",
            experiment="two_tone",
            quantity="output_pressure",
            domain="time",
            x=np.arange(3, dtype=float),
            values=np.arange(3, dtype=float)[:, None],
            x_unit="sample",
            value_unit="Pa",
            value_scale="linear",
            series=[comparison.SeriesDescriptor(0, "3 V RMS", vrms=3.0)],
            fs_hz=96000.0,
        )

    def make_time_comparison(self) -> comparison.TimeSeriesComparison:
        measured = np.asarray([1.0, -1.0, 0.5])
        simulated = np.asarray([1.1, -0.9, 0.4])
        error = simulated - measured
        centered_measured = measured - np.mean(measured)
        centered_simulated = simulated - np.mean(simulated)
        ac_error = centered_simulated - centered_measured
        return comparison.TimeSeriesComparison(
            time_s=np.arange(3, dtype=float) / 96000.0,
            reference=measured,
            simulation=simulated,
            lag_samples=10,
            descriptor=comparison.SeriesDescriptor(0, "3 V RMS", vrms=3.0),
            rmse_pa=float(np.sqrt(np.mean(error * error))),
            reference_rms_pa=float(np.sqrt(np.mean(measured * measured))),
            nrmse=float(np.linalg.norm(error) / np.linalg.norm(measured)),
            ac_rmse_pa=float(np.sqrt(np.mean(ac_error * ac_error))),
            ac_reference_rms_pa=float(np.sqrt(np.mean(centered_measured ** 2))),
            ac_nrmse=float(np.linalg.norm(ac_error) / np.linalg.norm(centered_measured)),
            fractional_offset_samples=0.25,
            alignment_correlation=0.998,
        )

    def make_result(self, reference, time_comparison, extra_data=None):
        data = {
            "solver_mode": "Schneller Python-Solver",
            "primary_solver_backend": "fast",
            "fast_solver_time_s": 0.01,
            "reference": reference,
            "reference_metadata": comparison.describe_reference(reference),
            "time_reference": reference,
            "time_reference_metadata": comparison.describe_reference(reference),
            "reference_time_comparison": time_comparison,
            "input": np.asarray([0.0, 1.0, 0.0]),
            "output": np.asarray([1.1, -0.9, 0.4]),
            "f1_hz": 81.4,
            "f2_hz": 344.79,
            "input_rms_voltage_v": 3.0,
            "input_level_definition": "A_in = rms(V_in)",
        }
        if extra_data:
            data.update(extra_data)
        return SimpleNamespace(
            experiment="two_tone",
            preset_name="Spk-2",
            fs=96000.0,
            data=data,
            summary="synthetischer Validierungsfall",
        )

    def test_time_package_contains_original_reference_tables_metrics_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = self.make_time_reference(root / "time_reference.txt")
            time_comparison = self.make_time_comparison()
            result = self.make_result(reference, time_comparison)
            config = {
                "experiment": "two_tone",
                "preset_name": "Spk-2",
                "solver_mode": "Schneller Python-Solver",
                "fs": 96000.0,
                "vrms": 3.0,
                "freq1_hz": 81.4,
                "freq2_hz": 344.79,
                "freq_hz": 999.0,
                "reference_path": str(reference.path),
                "reference_enabled": True,
            }

            package = validation_export.export_validation_package(
                result,
                config,
                root / "exports",
                created_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
            )

            self.assertTrue((package / "simulation_raw.npz").is_file())
            self.assertTrue((package / "time_comparison.csv").is_file())
            self.assertTrue((package / "metrics.csv").is_file())
            copied = package / "references" / reference.path.name
            self.assertEqual(copied.read_bytes(), reference.path.read_bytes())

            table = np.loadtxt(package / "time_comparison.csv", delimiter=",", skiprows=1)
            np.testing.assert_allclose(table[:, 1], result.data["reference_time_comparison"].reference)
            np.testing.assert_allclose(table[:, 2], result.data["reference_time_comparison"].simulation)
            np.testing.assert_allclose(table[:, 3], table[:, 2] - table[:, 1])

            metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(metadata["solver"]["primary_backend"], "fast")
            self.assertNotIn("freq_hz", metadata["run_config"])
            self.assertEqual(
                metadata["run_config"]["reference_path"], reference.path.name
            )
            self.assertEqual(
                metadata["reference_files"][0]["source_path"], reference.path.name
            )
            self.assertEqual(
                metadata["reference_metadata"]["path"], reference.path.name
            )
            self.assertEqual(
                metadata["result_metadata"]["reference_metadata"]["path"],
                reference.path.name,
            )
            self.assertEqual(metadata["validation_status"], "evaluated_without_acceptance_threshold")
            self.assertEqual(len(metadata["reference_files"][0]["sha256"]), 64)
            expected_level_difference = 20.0 * np.log10(
                np.std(time_comparison.simulation) / np.std(time_comparison.reference)
            )
            self.assertAlmostEqual(
                time_comparison.ac_level_difference_db,
                expected_level_difference,
            )
            metrics = {item["metric"]: item for item in metadata["metrics"]}
            self.assertAlmostEqual(
                float(metrics["ac_level_difference"]["value"]),
                expected_level_difference,
            )
            self.assertEqual(metrics["ac_level_difference"]["unit"], "dB")

    def test_psd_package_uses_reference_frequency_grid_and_copies_both_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            time_reference = self.make_time_reference(root / "time_reference.txt")
            psd_path = root / "psd_reference.txt"
            psd_path.write_text("0,1\n5,0.5\n10,0.25\n", encoding="utf-8")
            psd_reference = comparison.ReferenceData(
                path=psd_path,
                figure="7d",
                preset_name="Spk-2",
                speaker_model="FSN020.71-4",
                source_type="measurement",
                experiment="two_tone",
                quantity="output_pressure_psd",
                domain="frequency",
                x=np.asarray([0.0, 5.0, 10.0]),
                values=np.asarray([1.0, 0.5, 0.25]),
                x_unit="Hz",
                value_unit="Pa^2/Hz",
                value_scale="linear",
                expected_vrms=3.0,
            )
            spectrum = comparison.SpectrumComparison(
                frequency_hz=np.asarray([0.0, 5.0, 10.0]),
                reference_linear=np.asarray([1.0, 0.5, 0.25]),
                simulation_linear=np.asarray([0.9, 0.55, 0.2]),
                reference_db=np.asarray([0.0, -3.0102999566, -6.0205999133]),
                simulation_db=np.asarray([-0.4575749056, -2.5963731051, -6.9897000434]),
                difference_db=np.asarray([-0.4575749056, 0.4139268515, -0.9691001301]),
                relevant_mask=np.asarray([True, True, True]),
            )
            result = self.make_result(
                time_reference,
                self.make_time_comparison(),
                {
                    "reference": psd_reference,
                    "reference_metadata": comparison.describe_reference(psd_reference),
                    "time_reference": time_reference,
                    "time_reference_metadata": comparison.describe_reference(time_reference),
                    "reference_psd_comparison": spectrum,
                    "psd_relevant_dynamic_range_db": 80.0,
                },
            )
            config = {"experiment": "two_tone", "preset_name": "Spk-2", "vrms": 3.0}
            package = validation_export.export_validation_package(
                result,
                config,
                root / "exports",
                created_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
            )

            spectrum_table = np.loadtxt(
                package / "spectrum_comparison.csv", delimiter=",", skiprows=1
            )
            np.testing.assert_array_equal(spectrum_table[:, 0], spectrum.frequency_hz)
            np.testing.assert_allclose(spectrum_table[:, 1], spectrum.reference_linear)
            np.testing.assert_allclose(spectrum_table[:, 2], spectrum.simulation_linear)
            self.assertTrue((package / "references" / psd_path.name).is_file())
            self.assertTrue((package / "references" / time_reference.path.name).is_file())

            metrics_text = (package / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn("spectral_rmse", metrics_text)
            self.assertIn("ac_nrmse", metrics_text)
            self.assertIn("ac_level_difference", metrics_text)

    def test_rejects_results_without_fast_primary_backend(self):
        result = SimpleNamespace(
            data={
                "primary_solver_backend": "reference",
                "reference_metadata": {},
                "reference_time_comparison": object(),
            }
        )
        self.assertFalse(validation_export.can_export_validation_package(result))

    def test_case_id_uses_declared_level_with_sqrt2_hypothesis(self):
        result = SimpleNamespace(
            experiment="two_tone",
            preset_name="Spk-2",
            data={
                "f1_hz": 81.4,
                "f2_hz": 344.79,
                "input_level_parameter_vrms": 3.0,
                "input_rms_voltage_v": 3.0 * np.sqrt(2.0),
            },
        )

        case_id = validation_export.build_validation_case_id(result, {})

        self.assertIn("_3Vrms_", case_id)
        self.assertNotIn("4p242", case_id)

    def test_stationary_sine_config_omits_two_tone_fields(self):
        config = {
            "experiment": "stationary_sine",
            "preset_name": "Spk-1",
            "freq_hz": 28.75,
            "freq1_hz": 28.75,
            "freq2_hz": 121.78,
            "phase1_rad": 0.8472,
            "phase2_rad": 2.2179,
        }

        relevant = validation_export._relevant_run_config(config)

        self.assertEqual(relevant["freq_hz"], 28.75)
        for key in ("freq1_hz", "freq2_hz", "phase1_rad", "phase2_rad"):
            self.assertNotIn(key, relevant)


if __name__ == "__main__":
    unittest.main()
