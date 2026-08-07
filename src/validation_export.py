"""Reproduzierbarer Export eines einzelnen physikalischen Validierungsfalls.

Das Modul verwendet ausschliesslich bereits berechnete GUI-Ergebnisse. Es
veraendert weder Simulation noch Ausrichtung oder Spektralschaetzung. Die
Originalreferenzen werden unveraendert kopiert; zusaetzlich werden Messung und
Simulation auf gemeinsamen Rastern als transparente Texttabellen gespeichert.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np

import WDFSolverFast as wdf_fast
import WDFSolverVPres as wdf_reference


SCHEMA_VERSION = "1.0"


def can_export_validation_package(result: Any) -> bool:
    """Prueft, ob ein Ergebnis einen Fast-Solver-Messvergleich enthaelt."""
    if result is None:
        return False
    data = getattr(result, "data", {})
    return bool(
        data.get("primary_solver_backend") == "fast"
        and "reference_metadata" in data
        and (
            "reference_time_comparison" in data
            or "reference_psd_comparison" in data
        )
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Nicht JSON-serialisierbarer Wert: {type(value).__name__}")


def _number_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _safe_token(value: Any) -> str:
    token = str(value).strip().replace("_", "-")
    token = re.sub(r"[^A-Za-z0-9.-]+", "-", token)
    return token.strip("-.") or "Fall"


def build_validation_case_id(result: Any, run_config: dict[str, Any]) -> str:
    """Erzeugt eine kurze, in Dateinamen verwendbare Fallkennung."""
    data = result.data
    parts = [_safe_token(result.preset_name), _safe_token(result.experiment)]

    if result.experiment == "stationary_sine" and "freq_hz" in data:
        parts.append(f"{_number_token(data['freq_hz'])}Hz")
    elif result.experiment == "two_tone":
        if "f1_hz" in data and "f2_hz" in data:
            parts.append(
                f"{_number_token(data['f1_hz'])}-{_number_token(data['f2_hz'])}Hz"
            )
    elif result.experiment == "external_signal":
        source = Path(str(data.get("source_path", run_config.get("signal_path", "Signal"))))
        parts.append(_safe_token(source.stem))

    nominal_vrms = data.get("nominal_vrms", run_config.get("vrms"))
    if nominal_vrms is not None:
        parts.append(f"{_number_token(nominal_vrms)}Vrms")
    parts.append("Fast")
    return "_".join(parts)


def _create_unique_staging_directory(
    base_directory: Path,
    folder_stem: str,
) -> tuple[Path, Path]:
    base_directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index:03d}"
        target = base_directory / f"{folder_stem}{suffix}"
        staging = base_directory / f".{folder_stem}{suffix}.incomplete"
        if not target.exists() and not staging.exists():
            staging.mkdir()
            return staging, target
        index += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_table(path: Path, header: list[str], columns: list[np.ndarray]) -> None:
    arrays = [np.asarray(column) for column in columns]
    if not arrays or any(array.ndim != 1 for array in arrays):
        raise ValueError("Exporttabellen benoetigen eindimensionale Spalten.")
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("Alle Exportspalten muessen dieselbe Laenge besitzen.")
    matrix = np.column_stack(arrays)
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.17g",
    )


def _metric(
    rows: list[dict[str, Any]],
    name: str,
    value: Any,
    unit: str,
    scope: str,
    definition: str = "",
) -> None:
    rows.append(
        {
            "metric": name,
            "value": _jsonable(value),
            "unit": unit,
            "scope": scope,
            "definition": definition,
        }
    )


def _time_export(
    directory: Path,
    data: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> list[str]:
    comparison = data.get("reference_time_comparison")
    if comparison is None:
        return []

    error = np.asarray(comparison.simulation) - np.asarray(comparison.reference)
    filename = "time_comparison.csv"
    _write_table(
        directory / filename,
        [
            "time_s",
            "measurement_pressure_pa",
            "simulation_pressure_pa",
            "simulation_minus_measurement_pa",
        ],
        [comparison.time_s, comparison.reference, comparison.simulation, error],
    )

    _metric(metrics, "rmse", comparison.rmse_pa, "Pa", "time")
    _metric(metrics, "reference_rms", comparison.reference_rms_pa, "Pa", "time")
    _metric(
        metrics,
        "absolute_nrmse",
        comparison.nrmse,
        "1",
        "time",
        "||P_sim-P_meas||_2 / ||P_meas||_2",
    )
    _metric(metrics, "ac_rmse", comparison.ac_rmse_pa, "Pa", "time")
    _metric(metrics, "ac_reference_rms", comparison.ac_reference_rms_pa, "Pa", "time")
    _metric(
        metrics,
        "ac_nrmse",
        comparison.ac_nrmse,
        "1",
        "time",
        "||AC(P_sim)-AC(P_meas)||_2 / ||AC(P_meas)||_2",
    )
    _metric(metrics, "alignment_integer_start", comparison.lag_samples, "samples", "time")
    _metric(
        metrics,
        "alignment_fractional_offset",
        comparison.fractional_offset_samples,
        "samples",
        "time",
    )
    _metric(
        metrics,
        "alignment_start",
        comparison.alignment_start_samples,
        "samples",
        "time",
    )
    _metric(metrics, "alignment_correlation", comparison.alignment_correlation, "1", "time")
    _metric(metrics, "comparison_samples", comparison.comparison_samples, "samples", "time")
    _metric(metrics, "comparison_duration", comparison.duration_s, "s", "time")
    return [filename]


def _spectrum_export(
    directory: Path,
    data: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> list[str]:
    spectrum = data.get("reference_psd_comparison")
    if spectrum is None:
        return []

    filename = "spectrum_comparison.csv"
    _write_table(
        directory / filename,
        [
            "frequency_hz",
            "measurement_psd_pa2_per_hz",
            "simulation_psd_pa2_per_hz",
            "measurement_psd_db_per_hz",
            "simulation_psd_db_per_hz",
            "simulation_minus_measurement_db",
            "relevant_frequency_point",
        ],
        [
            spectrum.frequency_hz,
            spectrum.reference_linear,
            spectrum.simulation_linear,
            spectrum.reference_db,
            spectrum.simulation_db,
            spectrum.difference_db,
            np.asarray(spectrum.relevant_mask, dtype=np.int8),
        ],
    )

    finite_relevant = (
        np.asarray(spectrum.relevant_mask, dtype=bool)
        & np.isfinite(spectrum.difference_db)
    )
    relevant_difference = np.asarray(spectrum.difference_db)[finite_relevant]
    _metric(metrics, "frequency_points", len(spectrum.frequency_hz), "points", "spectrum")
    _metric(metrics, "relevant_frequency_points", len(relevant_difference), "points", "spectrum")
    _metric(
        metrics,
        "relevant_dynamic_range",
        data.get("psd_relevant_dynamic_range_db", 80.0),
        "dB",
        "spectrum",
        "Frequenzpunkte mit Referenz-PSD innerhalb dieses Bereichs unterhalb des Maximums",
    )
    if relevant_difference.size:
        _metric(
            metrics,
            "spectral_rmse",
            float(np.sqrt(np.mean(relevant_difference * relevant_difference))),
            "dB",
            "spectrum",
        )
        _metric(
            metrics,
            "spectral_mean_difference",
            float(np.mean(relevant_difference)),
            "dB",
            "spectrum",
        )
        _metric(
            metrics,
            "spectral_maximum_absolute_difference",
            float(np.max(np.abs(relevant_difference))),
            "dB",
            "spectrum",
        )
    return [filename]


def _copy_references(directory: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    references_directory = directory / "references"
    references_directory.mkdir()
    candidates = [
        ("selected_reference", data.get("reference")),
        ("time_reference", data.get("time_reference")),
    ]
    copied: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for role, reference in candidates:
        source_value = getattr(reference, "path", None)
        if source_value is None:
            continue
        source = Path(source_value).expanduser().resolve()
        if source in seen:
            for item in copied:
                if Path(item["source_path"]).resolve() == source:
                    item["roles"].append(role)
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Referenzdatei nicht gefunden: {source}")

        destination = references_directory / source.name
        if destination.exists():
            destination = references_directory / f"{role}_{source.name}"
        shutil.copy2(source, destination)
        copied.append(
            {
                "roles": [role],
                "source_path": str(source),
                "archive_path": destination.relative_to(directory).as_posix(),
                "sha256": _sha256(destination),
            }
        )
        seen.add(source)
    if not copied:
        raise ValueError("Im Ergebnis ist keine kopierbare Referenzdatei enthalten.")
    return copied


def _result_scalar_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, np.generic)) or value is None:
            metadata[key] = _jsonable(value)
        elif key in {
            "harmonic_data",
            "reference_metadata",
            "time_reference_metadata",
        }:
            metadata[key] = _jsonable(value)
    return metadata


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "value", "unit", "scope", "definition"),
        )
        writer.writeheader()
        writer.writerows(rows)


def export_validation_package(
    result: Any,
    run_config: dict[str, Any],
    base_directory: str | Path,
    figure: Any | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Schreibt einen vollstaendigen, kollisionsfreien Validierungsfall."""
    if not can_export_validation_package(result):
        raise ValueError(
            "Ein Validierungspaket erfordert einen Messvergleich mit dem schnellen Solver."
        )
    if not isinstance(run_config, dict) or not run_config:
        raise ValueError("Die beim Simulationslauf verwendete Konfiguration fehlt.")

    timestamp = created_at or datetime.now().astimezone()
    case_id = build_validation_case_id(result, run_config)
    folder_stem = f"{timestamp:%Y-%m-%d}_{case_id}"
    staging, target = _create_unique_staging_directory(
        Path(base_directory).expanduser().resolve(), folder_stem
    )

    try:
        data = result.data
        metrics: list[dict[str, Any]] = []
        produced_files = []

        arrays = {
            key: value
            for key, value in data.items()
            if isinstance(value, np.ndarray) and value.dtype != object
        }
        arrays["preset_name"] = np.asarray(result.preset_name)
        arrays["experiment"] = np.asarray(result.experiment)
        arrays["fs_hz"] = np.asarray(np.nan if result.fs is None else result.fs)
        np.savez_compressed(staging / "simulation_raw.npz", **arrays)
        produced_files.append("simulation_raw.npz")

        produced_files.extend(_time_export(staging, data, metrics))
        produced_files.extend(_spectrum_export(staging, data, metrics))
        _write_metrics(staging / "metrics.csv", metrics)
        produced_files.append("metrics.csv")

        reference_copies = _copy_references(staging, data)
        produced_files.extend(item["archive_path"] for item in reference_copies)

        if figure is not None:
            figure.savefig(staging / "comparison_plot.pdf", bbox_inches="tight")
            produced_files.append("comparison_plot.pdf")

        model_parameters = wdf_reference.BERNARDINI_PRESETS.get(result.preset_name, {})
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": timestamp.astimezone().isoformat(timespec="seconds"),
            "case_id": case_id,
            "validation_status": "evaluated_without_acceptance_threshold",
            "experiment": result.experiment,
            "preset_name": result.preset_name,
            "fs_hz": result.fs,
            "summary": result.summary,
            "run_config": _jsonable(run_config),
            "model_parameters": _jsonable(model_parameters),
            "solver": {
                "mode": data.get("solver_mode"),
                "primary_backend": data.get("primary_solver_backend"),
                "fast_solver_version": getattr(wdf_fast, "__version__", None),
                "reference_solver_version": getattr(wdf_reference, "__version__", None),
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "python_executable": sys.executable,
            },
            "reference_files": reference_copies,
            "reference_metadata": _jsonable(data.get("reference_metadata", {})),
            "time_reference_metadata": _jsonable(data.get("time_reference_metadata", {})),
            "metrics": metrics,
            "result_metadata": _result_scalar_metadata(data),
            "array_keys": sorted(arrays),
            "files": sorted(produced_files + ["metadata.json"]),
            "notes": [
                "Originalreferenzen wurden bytegetreu kopiert und mit SHA-256 dokumentiert.",
                "Zeitdaten sind ausschliesslich zeitlich ausgerichtet; es erfolgt keine multiplikative Pegelanpassung.",
                "Spektraldaten liegen auf dem Frequenzraster der ausgewaehlten Referenzdatei.",
                "Es wurde kein automatisches Bestehenskriterium angewendet.",
            ],
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
