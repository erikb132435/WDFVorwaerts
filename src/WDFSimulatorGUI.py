"""Grafische Bedienoberflaeche fuer den Bernardini-WDF-Simulator.

Die GUI trennt die Definition einer Untersuchung von ihrer Darstellung. Der
numerische Solver in ``WDFSolverVPres.py`` bleibt dabei unveraendert und dient
weiterhin als Referenzimplementierung.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import wavfile

import WDFSolverVPres as wdf
import spectral_comparison as comparison


EXPERIMENTS = {
    "Kleinsignal-Frequenzgänge (Referenzdaten)": "small_signal_reference",
    "Auslenkungsabhängige Modellparameter": "nonlinear_characteristics",
    "Stationäre Sinusanregung": "stationary_sine",
    "THD-Frequenzgang aus stationären Sinustönen": "thd_frequency_response",
    "Zweiton-Anregung": "two_tone",
    "Exponentieller Sinus-Sweep": "exponential_sweep",
    "Externes Signal (Musik / WAV / NPY)": "external_signal",
}

PLOT_OPTIONS = {
    "small_signal_reference": [
        "Elektrische Eingangsimpedanz und Ausgangsdruck",
        "Elektrische Eingangsimpedanz",
        "Ausgangsdruck-Frequenzgang",
    ],
    "nonlinear_characteristics": ["Nichtlineare Kennlinien"],
    "stationary_sine": [
        "Zeitverlauf",
        "Leistungsdichtespektrum",
        "Leistungsspektrum",
        "Harmonische Komponenten",
        "Membranauslenkung",
        "Membrangeschwindigkeit",
        "Schwingspulenstrom",
    ],
    "thd_frequency_response": ["THD über der Frequenz"],
    "two_tone": [
        "Zeitverlauf", "Leistungsdichtespektrum", "Leistungsspektrum",
        "Membranauslenkung", "Membrangeschwindigkeit", "Schwingspulenstrom",
    ],
    "exponential_sweep": [
        "Zeitverlauf", "Leistungsdichtespektrum", "Membranauslenkung",
        "Membrangeschwindigkeit", "Schwingspulenstrom",
    ],
    "external_signal": [
        "Zeitverlauf", "Leistungsdichtespektrum", "Leistungsspektrum",
        "Membranauslenkung", "Membrangeschwindigkeit", "Schwingspulenstrom",
    ],
}

STANDARD_INPUT_LEVEL_MODE = "Physikalischer RMS-Wert"
BERNARDINI_INPUT_LEVEL_MODE = "Bernardini-Reproduktion (A_in * sqrt(2))"
INPUT_LEVEL_MODES = {
    STANDARD_INPUT_LEVEL_MODE: 1.0,
    BERNARDINI_INPUT_LEVEL_MODE: float(np.sqrt(2.0)),
}


@dataclass
class SimulationResult:
    """Ein GUI-unabhaengiges Ergebnis einer Untersuchung."""

    experiment: str
    preset_name: str
    fs: float | None
    data: dict[str, Any]
    summary: str


def _as_mono_float(data: np.ndarray) -> np.ndarray:
    """Konvertiert WAV-Daten in ein mono Float-Signal mit sinnvoller Skalierung."""
    source = np.asarray(data)
    integer_source = np.issubdtype(source.dtype, np.integer)
    if integer_source:
        info = np.iinfo(source.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
    else:
        scale = 1.0
    if source.ndim > 1:
        source = np.mean(source.astype(float), axis=1)
    if integer_source:
        return source.astype(float) / scale
    return source.astype(float)


def _normalise_rms(signal: np.ndarray, target_vrms: float) -> np.ndarray:
    """Skaliert ein Signal auf den vorgegebenen RMS-Spannungspegel."""
    signal = np.asarray(signal, dtype=float)
    rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
    if rms <= 1e-18:
        raise ValueError("Das Eingangssignal besitzt keinen von null verschiedenen RMS-Wert.")
    return signal * (float(target_vrms) / rms)


def _resolve_input_level(config: dict[str, Any]) -> tuple[float, float, float, str]:
    """Uebersetzt den angezeigten A_in-Wert in den tatsaechlichen RMS-Antrieb."""
    nominal_vrms = float(config["vrms"])
    if not np.isfinite(nominal_vrms) or nominal_vrms <= 0.0:
        raise ValueError("Der Eingangspegel A_in muss positiv und endlich sein.")
    mode = str(config.get("input_level_mode", STANDARD_INPUT_LEVEL_MODE))
    if mode not in INPUT_LEVEL_MODES:
        raise ValueError(f"Unbekannte Pegelkonvention: {mode}")
    scale_factor = INPUT_LEVEL_MODES[mode]
    effective_vrms = nominal_vrms * scale_factor
    return nominal_vrms, effective_vrms, scale_factor, mode


def _input_level_metadata(
    nominal_vrms: float,
    effective_vrms: float,
    scale_factor: float,
    mode: str,
) -> dict[str, Any]:
    return {
        "input_level_mode": mode,
        "input_level_scale_factor": scale_factor,
        "nominal_input_level_vrms": nominal_vrms,
        "effective_input_level_vrms": effective_vrms,
    }


def _input_level_summary(nominal_vrms: float, effective_vrms: float, scale_factor: float) -> str:
    if np.isclose(scale_factor, 1.0):
        return f"{nominal_vrms:g} V RMS"
    return f"A_in = {nominal_vrms:g} V, effektiver RMS-Pegel = {effective_vrms:.6g} V"


def _simulate(
    fs: float,
    params: dict[str, Any],
    signal: np.ndarray,
    record_internal: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fuehrt den Referenzsolver optional mit interner Zustandsaufzeichnung aus."""
    model = wdf.BernardiniSpeakerSolver(fs, params)
    simulation = wdf.simulate_model(
        model,
        np.asarray(signal, dtype=float),
        record_traces=record_internal,
    )
    if not record_internal:
        return simulation, {}
    output = simulation["output_pressure"]
    traces = {key: value for key, value in simulation.items() if key != "output_pressure"}
    return output, traces


def _load_optional_reference(
    config: dict[str, Any],
    preset_name: str,
    expected_experiment: str | None = None,
    required: bool = False,
) -> comparison.ReferenceData | None:
    """Laedt eine aktivierte Referenz und prueft ihre Versuchszuordnung."""
    enabled = bool(config.get("reference_enabled", False)) or required
    if not enabled:
        return None
    path = str(config.get("reference_path", "")).strip()
    if not path:
        raise ValueError("Bitte eine Bernardini-Referenzdatei auswählen.")
    reference = comparison.load_reference_data(path)
    if reference.preset_name != preset_name:
        raise ValueError(
            f"Referenzdatei gehört zu {reference.preset_name}, ausgewählt ist {preset_name}."
        )
    if expected_experiment is not None and reference.experiment != expected_experiment:
        raise ValueError(
            f"Referenzdatei gehört zur Untersuchung '{reference.experiment}', erwartet wird "
            f"'{expected_experiment}'."
        )
    return reference


def _reference_comparison_data(
    reference: comparison.ReferenceData | None,
    output: np.ndarray,
    fs: float,
    vrms: float,
    settle_s: float,
    frequencies_hz: tuple[float, ...] = (),
    phases_rad: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Erzeugt je nach Referenztyp einen Zeit- oder PSD-Vergleich."""
    if reference is None:
        return {}
    data: dict[str, Any] = {
        "reference": reference,
        "reference_metadata": comparison.describe_reference(reference),
    }
    notices: list[str] = []

    def add_time_comparison(time_reference: comparison.ReferenceData) -> None:
        comparison.validate_reference_excitation(
            time_reference, vrms, frequencies_hz, phases_rad
        )
        time_result = comparison.align_time_series(
            output,
            fs,
            time_reference,
            vrms=vrms,
            frequency_hz=frequencies_hz[0] if len(frequencies_hz) == 1 else None,
            simulation_start_s=settle_s,
        )
        data["time_reference"] = time_reference
        data["time_reference_metadata"] = comparison.describe_reference(time_reference)
        data["reference_time_comparison"] = time_result
        data["reference_time_s"] = time_result.time_s
        data["reference_output_pressure"] = time_result.reference
        data["simulation_output_pressure_aligned"] = time_result.simulation
        data["time_rmse_pa"] = time_result.rmse_pa
        data["time_reference_rms_pa"] = time_result.reference_rms_pa
        data["time_nrmse_relative_rms"] = time_result.nrmse
        data["time_ac_rmse_pa"] = time_result.ac_rmse_pa
        data["time_ac_reference_rms_pa"] = time_result.ac_reference_rms_pa
        data["time_ac_nrmse_relative_rms"] = time_result.ac_nrmse
        data["time_comparison_samples"] = time_result.comparison_samples
        data["time_comparison_duration_s"] = time_result.duration_s
        data["alignment_start_integer_sample"] = time_result.lag_samples
        data["alignment_fractional_offset_samples"] = time_result.fractional_offset_samples
        data["alignment_start_sample"] = time_result.alignment_start_samples
        data["alignment_start_s"] = time_result.alignment_start_samples / fs
        data["alignment_correlation"] = time_result.alignment_correlation
        data["nrmse_definition"] = "||P_sim - P_meas||_2 / ||P_meas||_2"
        data["ac_nrmse_definition"] = (
            "||(P_sim-mean(P_sim)) - (P_meas-mean(P_meas))||_2 / "
            "||P_meas-mean(P_meas)||_2"
        )
        notices.append(
            f"Zeitreferenz Fig. {time_reference.figure}, Spalte „{time_result.descriptor.label}“; "
            "der Simulationsausschnitt wurde per normierter Kreuzkorrelation und "
            "Subsample-Korrektur ausschließlich zeitlich ausgerichtet. "
            f"Absoluter NRMSE = {time_result.nrmse:.6g} "
            f"({100.0 * time_result.nrmse:.3f} %); AC-NRMSE = "
            f"{time_result.ac_nrmse:.6g} ({100.0 * time_result.ac_nrmse:.3f} %); "
            f"N = {time_result.comparison_samples}, "
            f"Simulationsstart = {time_result.alignment_start_samples:.3f} Samples "
            f"({1000.0 * time_result.alignment_start_samples / fs:.3f} ms)."
        )

    comparison.validate_reference_excitation(reference, vrms, frequencies_hz, phases_rad)
    if reference.quantity == "output_pressure_psd":
        start = max(0, int(round(float(settle_s) * fs)))
        reference_analysis = np.asarray(output[start:], dtype=float)
        spectral = comparison.compare_psd(
            reference_analysis,
            fs,
            reference,
        )
        data["reference_psd_comparison"] = spectral
        data["reference_frequency_hz"] = spectral.frequency_hz
        data["reference_psd_linear"] = spectral.reference_linear
        data["simulation_psd_linear_on_reference_grid"] = spectral.simulation_linear
        data["psd_difference_db"] = spectral.difference_db
        confirmed = "; ".join(reference.psd_config.confirmed_properties)
        assumptions = "; ".join(reference.psd_config.assumptions)
        normalization = ""
        if not np.isclose(reference.applied_value_scale_factor, 1.0):
            normalization = (
                f" Referenzwerte wurden mit Faktor {reference.applied_value_scale_factor:g} "
                "auf Pa²/Hz normiert."
            )
        notices.append(
            f"Referenz Fig. {reference.figure}: identisches Frequenzraster. "
            f"Bestätigt: {confirmed}. Noch unbestätigt: {assumptions}.{normalization}"
        )
        time_path = comparison.find_time_domain_companion(reference)
        if time_path is not None:
            add_time_comparison(comparison.load_reference_data(time_path))
        else:
            notices.append(
                "Keine zugehörige Zeitmessung gefunden; ein Zeitbereichs-NRMSE ist nicht definiert."
            )
    elif reference.quantity == "output_pressure":
        add_time_comparison(reference)
    else:
        raise ValueError("Der gewählte Referenzdatentyp passt nicht zu einer Zeitsimulation.")
    data["reference_notice"] = " ".join(notices)
    return data


def run_experiment(config: dict[str, Any]) -> SimulationResult:
    """Fuehrt eine Untersuchung ohne Abhaengigkeit von Tkinter aus.

    Diese Funktion kann spaeter auch fuer automatisierte Verifikationstests und
    Batch-Laeufe verwendet werden.
    """
    experiment = str(config["experiment"])
    preset_name = str(config["preset_name"])
    if preset_name not in wdf.BERNARDINI_PRESETS:
        raise ValueError(f"Unbekanntes Lautsprecherpreset: {preset_name}")
    params = dict(wdf.BERNARDINI_PRESETS[preset_name])

    if experiment == "small_signal_reference":
        primary = _load_optional_reference(config, preset_name, required=True)
        if primary.experiment != "small_signal":
            raise ValueError("Für Kleinsignal-Frequenzgänge wird eine linear_P- oder linear_Ze-Datei benötigt.")
        references = {primary.quantity: primary}
        other_quantity = (
            "electric_impedance_magnitude"
            if primary.quantity == "output_pressure_magnitude"
            else "output_pressure_magnitude"
        )
        companion_path = comparison.find_companion_reference(primary, other_quantity)
        if companion_path is not None:
            companion = comparison.load_reference_data(companion_path)
            references[companion.quantity] = companion
        stored_arrays: dict[str, np.ndarray] = {}
        if "electric_impedance_magnitude" in references:
            impedance = references["electric_impedance_magnitude"]
            stored_arrays["reference_impedance_frequency_hz"] = impedance.x
            stored_arrays["reference_impedance_magnitude_ohm"] = impedance.values
        if "output_pressure_magnitude" in references:
            pressure = references["output_pressure_magnitude"]
            stored_arrays["reference_pressure_frequency_hz"] = pressure.x
            stored_arrays["reference_pressure_magnitude_pa"] = pressure.values
        return SimulationResult(
            experiment,
            preset_name,
            None,
            {"small_signal_references": references, **stored_arrays},
            (
                f"{preset_name}: Kleinsignal-Referenzdaten aus Fig. {primary.figure}; "
                f"{len(references)} Messgröße(n) geladen."
            ),
        )

    if experiment == "nonlinear_characteristics":
        x_min_mm = float(config["x_min_mm"])
        x_max_mm = float(config["x_max_mm"])
        points = int(config["points"])
        if x_max_mm <= x_min_mm:
            raise ValueError("Das Maximum der Auslenkung muss größer als das Minimum sein.")
        if points < 32:
            raise ValueError("Für die Kennlinien werden mindestens 32 Stützstellen benötigt.")

        x_mm = np.linspace(x_min_mm, x_max_mm, points)
        x_m = x_mm * 1e-3
        bl = np.polyval(np.asarray(params["Bl_coeffs"])[::-1], x_m)
        le_mh = 1e3 * np.polyval(np.asarray(params["Le_coeffs"])[::-1], x_m)
        kms_n_per_mm = 1e-3 * np.polyval(np.asarray(params["Kms_coeffs"])[::-1], x_m)
        return SimulationResult(
            experiment,
            preset_name,
            None,
            {"x_mm": x_mm, "bl": bl, "le_mh": le_mh, "kms_n_per_mm": kms_n_per_mm},
            f"Kennlinien von {preset_name} für {x_min_mm:g} mm ≤ x ≤ {x_max_mm:g} mm.",
        )

    fs = float(config["fs"])
    if fs <= 0.0:
        raise ValueError("Die Abtastrate muss positiv sein.")
    nominal_vrms, effective_vrms, input_scale, input_level_mode = _resolve_input_level(config)
    input_level_data = _input_level_metadata(
        nominal_vrms, effective_vrms, input_scale, input_level_mode
    )

    if experiment == "stationary_sine":
        vrms = nominal_vrms
        duration_s = float(config["duration_s"])
        settle_s = float(config["settle_s"])
        frequency_mode = str(config["frequency_mode"])
        resonance_hz = wdf.estimate_small_signal_resonance(params)
        if frequency_mode == "f_r / 2":
            freq_hz = 0.5 * resonance_hz
        elif frequency_mode == "2 f_r":
            freq_hz = 2.0 * resonance_hz
        else:
            freq_hz = float(config["freq_hz"])
        if not 0.0 < freq_hz < fs / 2.0:
            raise ValueError("Die Sinusfrequenz muss zwischen 0 Hz und der Nyquist-Frequenz liegen.")
        if not 0.0 <= settle_s < duration_s:
            raise ValueError("Die Einschwingzeit muss kleiner als die Signaldauer sein.")

        signal = wdf.generate_sine(fs, duration_s, freq_hz, effective_vrms)
        output, traces = _simulate(fs, params, signal, bool(config.get("record_internal", False)))
        analysis = wdf.prepare_analysis_signal(output, fs, settle_s, [freq_hz])
        harmonic_data, harmonic_freqs, harmonic_power_db = wdf.measure_harmonic_levels_db(
            analysis, fs, freq_hz, harmonics=int(config["harmonics"])
        )
        thd_point = wdf.compute_output_thd_point(
            output, fs, freq_hz, harmonics=int(config["harmonics"]), settle_s=settle_s
        )
        reference = _load_optional_reference(config, preset_name, expected_experiment="sinusoid")
        reference_data = _reference_comparison_data(
            reference, output, fs, vrms, settle_s, frequencies_hz=(freq_hz,)
        )
        return SimulationResult(
            experiment,
            preset_name,
            fs,
            dict({
                "input": signal,
                "output": output,
                "analysis": analysis,
                "freq_hz": freq_hz,
                "resonance_hz": resonance_hz,
                "settle_s": settle_s,
                "harmonic_data": harmonic_data,
                "harmonic_freqs": harmonic_freqs,
                "harmonic_power_db": harmonic_power_db,
                "thd_db": thd_point["thd_db"],
            }, **input_level_data, **traces, **reference_data),
            (
                f"{preset_name}: stationärer Sinus bei {freq_hz:.3f} Hz, "
                f"{_input_level_summary(vrms, effective_vrms, input_scale)}; "
                f"verwendete Resonanzfrequenz {resonance_hz:.3f} Hz; THD {thd_point['thd_db']:.3f} dB."
            ),
        )

    if experiment == "thd_frequency_response":
        f_min = float(config["f_min_hz"])
        f_max = float(config["f_max_hz"])
        if not 0.0 < f_min < f_max < fs / 2.0:
            raise ValueError("Der Frequenzbereich muss innerhalb (0, Nyquist) liegen.")
        resonance_hz = wdf.estimate_small_signal_resonance(params)
        freqs, thd_db, dominant_harmonics, dominant_relative_db = (
            wdf.compute_stationary_sine_thd_curve(
                fs,
                params,
                effective_vrms,
                f_min,
                f_max,
                num_points=int(config["frequency_points"]),
                harmonics=int(config["harmonics"]),
                include_freqs_hz=[0.5 * resonance_hz, 2.0 * resonance_hz],
            )
        )
        return SimulationResult(
            experiment,
            preset_name,
            fs,
            {
                "frequencies_hz": freqs,
                "thd_db": thd_db,
                "dominant_harmonics": dominant_harmonics,
                "dominant_relative_db": dominant_relative_db,
                **input_level_data,
            },
            (
                f"{preset_name}: THD aus {len(freqs)} stationären Sinusmessungen "
                f"von {f_min:g} Hz bis {f_max:g} Hz; "
                f"{_input_level_summary(nominal_vrms, effective_vrms, input_scale)}."
            ),
        )

    if experiment == "two_tone":
        duration_s = float(config["duration_s"])
        settle_s = float(config["settle_s"])
        f1 = float(config["freq1_hz"])
        f2 = float(config["freq2_hz"])
        if not 0.0 < f1 < fs / 2.0 or not 0.0 < f2 < fs / 2.0:
            raise ValueError("Beide Frequenzen müssen innerhalb (0, Nyquist) liegen.")
        if not 0.0 <= settle_s < duration_s:
            raise ValueError("Die Einschwingzeit muss kleiner als die Signaldauer sein.")
        signal = wdf.generate_two_tone(
            fs,
            duration_s,
            f1,
            f2,
            effective_vrms,
            phase1_rad=float(config["phase1_rad"]),
            phase2_rad=float(config["phase2_rad"]),
        )
        output, traces = _simulate(fs, params, signal, bool(config.get("record_internal", False)))
        analysis = wdf.prepare_two_tone_analysis_signal(output, fs, settle_s)
        reference = _load_optional_reference(config, preset_name, expected_experiment="two_tone")
        reference_data = _reference_comparison_data(
            reference,
            output,
            fs,
            nominal_vrms,
            settle_s,
            frequencies_hz=(f1, f2),
            phases_rad=(float(config["phase1_rad"]), float(config["phase2_rad"])),
        )
        return SimulationResult(
            experiment,
            preset_name,
            fs,
            dict(
                {
                    "input": signal,
                    "output": output,
                    "analysis": analysis,
                    "f1_hz": f1,
                    "f2_hz": f2,
                    **input_level_data,
                },
                **traces,
                **reference_data,
            ),
            (
                f"{preset_name}: Zweiton-Anregung bei {f1:.3f} Hz und {f2:.3f} Hz; "
                f"{_input_level_summary(nominal_vrms, effective_vrms, input_scale)}."
            ),
        )

    if experiment == "exponential_sweep":
        duration_s = float(config["duration_s"])
        f_min = float(config["f_min_hz"])
        f_max = float(config["f_max_hz"])
        if not 0.0 < f_min < f_max < fs / 2.0:
            raise ValueError("Der Sweep-Bereich muss innerhalb (0, Nyquist) liegen.")
        signal, inverse = wdf.generate_sweep(
            fs, duration_s, f_min, f_max, effective_vrms
        )
        output, traces = _simulate(fs, params, signal, bool(config.get("record_internal", False)))
        analysis = wdf.prepare_analysis_signal(output, fs)
        return SimulationResult(
            experiment,
            preset_name,
            fs,
            dict(
                {
                    "input": signal,
                    "inverse": inverse,
                    "output": output,
                    "analysis": analysis,
                    **input_level_data,
                },
                **traces,
            ),
            (
                f"{preset_name}: exponentieller Sinus-Sweep von {f_min:g} Hz bis {f_max:g} Hz; "
                f"{_input_level_summary(nominal_vrms, effective_vrms, input_scale)}."
            ),
        )

    if experiment == "external_signal":
        source_path = Path(str(config["signal_path"])).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Eingangssignal nicht gefunden: {source_path}")
        if source_path.suffix.lower() == ".wav":
            file_fs, source = wavfile.read(source_path)
            signal = _as_mono_float(source)
            fs = float(file_fs)
        elif source_path.suffix.lower() == ".npy":
            signal = np.asarray(np.load(source_path), dtype=float).squeeze()
            if signal.ndim != 1:
                raise ValueError("Die NPY-Datei muss ein eindimensionales Signal enthalten.")
        else:
            raise ValueError("Unterstützt werden WAV- und NPY-Dateien.")
        if signal.size == 0:
            raise ValueError("Die Signaldatei ist leer.")
        if bool(config["normalise_external"]):
            signal = _normalise_rms(signal, effective_vrms)
        output, traces = _simulate(fs, params, signal, bool(config.get("record_internal", False)))
        settle_s = float(config["settle_s"])
        if not 0.0 <= settle_s < len(signal) / fs:
            raise ValueError("Die Einschwingzeit muss kleiner als die Dauer des externen Signals sein.")
        analysis = wdf.prepare_analysis_signal(output, fs, settle_s)
        reference = _load_optional_reference(config, preset_name, expected_experiment="music")
        reference_data = _reference_comparison_data(
            reference, output, fs, nominal_vrms, settle_s
        )
        return SimulationResult(
            experiment,
            preset_name,
            fs,
            dict(
                {
                    "input": signal,
                    "output": output,
                    "analysis": analysis,
                    "source_path": str(source_path),
                    **input_level_data,
                },
                **traces,
                **reference_data,
            ),
            (
                f"{preset_name}: externes Signal „{source_path.name}“, {fs:g} Hz, "
                f"{len(signal) / fs:.3f} s."
            ),
        )

    raise ValueError(f"Nicht implementierte Untersuchung: {experiment}")


class WDFSimulatorGUI(tk.Tk):
    """Tkinter-Hauptfenster fuer strukturierte WDF-Untersuchungen."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Bernardini-WDF – Simulationsumgebung")
        self.geometry("1320x820")
        self.minsize(1050, 680)

        self._messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self.result: SimulationResult | None = None
        self._parameter_rows: dict[str, tuple[ttk.Label, tk.Widget]] = {}

        self._create_variables()
        self._create_layout()
        self._experiment_changed()
        self.after(100, self._poll_worker)

    def _create_variables(self) -> None:
        self.experiment_label = tk.StringVar(value="Stationäre Sinusanregung")
        self.preset_name = tk.StringVar(value="Spk-1")
        self.plot_kind = tk.StringVar(value="Zeitverlauf")
        self.fs = tk.StringVar(value="48000")
        self.vrms = tk.StringVar(value="11.5")
        self.input_level_mode = tk.StringVar(value=STANDARD_INPUT_LEVEL_MODE)
        self.duration_s = tk.StringVar(value="1.0")
        self.settle_s = tk.StringVar(value="0.2")
        self.frequency_mode = tk.StringVar(value="frei wählbar")
        self.freq_hz = tk.StringVar(value="50.0")
        self.freq1_hz = tk.StringVar(value="28.75")
        self.freq2_hz = tk.StringVar(value="121.78")
        self.phase1_rad = tk.StringVar(value="0.8472")
        self.phase2_rad = tk.StringVar(value="2.2179")
        self.f_min_hz = tk.StringVar(value="20.0")
        self.f_max_hz = tk.StringVar(value="20000.0")
        self.frequency_points = tk.StringVar(value="24")
        self.harmonics = tk.StringVar(value="5")
        self.x_min_mm = tk.StringVar(value="-5.0")
        self.x_max_mm = tk.StringVar(value="5.0")
        self.points = tk.StringVar(value="801")
        self.signal_path = tk.StringVar(value="")
        self.normalise_external = tk.BooleanVar(value=True)
        self.record_internal = tk.BooleanVar(value=False)
        self.reference_enabled = tk.BooleanVar(value=False)
        self.reference_path = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Bereit")
        self.summary_text = tk.StringVar(value="Noch keine Untersuchung ausgeführt.")

    def _create_layout(self) -> None:
        outer = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = ttk.Frame(outer, padding=8)
        plot_area = ttk.Frame(outer, padding=4)
        outer.add(controls, weight=0)
        outer.add(plot_area, weight=1)

        ttk.Label(controls, text="Untersuchung", font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        experiment_box = ttk.Combobox(
            controls,
            textvariable=self.experiment_label,
            values=list(EXPERIMENTS),
            state="readonly",
            width=43,
        )
        experiment_box.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        experiment_box.bind("<<ComboboxSelected>>", lambda _event: self._experiment_changed())

        ttk.Label(controls, text="Lautsprechermodell").grid(row=2, column=0, sticky="w", pady=3)
        preset_box = ttk.Combobox(
            controls,
            textvariable=self.preset_name,
            values=list(wdf.BERNARDINI_PRESETS),
            state="readonly",
            width=18,
        )
        preset_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        preset_box.bind("<<ComboboxSelected>>", lambda _event: self._preset_changed())

        ttk.Separator(controls).grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(controls, text="Versuchsparameter", font=("TkDefaultFont", 11, "bold")).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )

        params = [
            ("fs", "Abtastrate F_s [Hz]", self.fs, None),
            ("vrms", "Nennpegel A_in [V]", self.vrms, None),
            (
                "input_level_mode",
                "Pegelkonvention",
                self.input_level_mode,
                list(INPUT_LEVEL_MODES),
            ),
            ("duration_s", "Signaldauer [s]", self.duration_s, None),
            ("settle_s", "Einschwingzeit [s]", self.settle_s, None),
            (
                "frequency_mode",
                "Sinusfrequenz",
                self.frequency_mode,
                ["frei wählbar", "f_r / 2", "2 f_r"],
            ),
            ("freq_hz", "f_in [Hz]", self.freq_hz, None),
            ("freq1_hz", "f_in,1 [Hz]", self.freq1_hz, None),
            ("freq2_hz", "f_in,2 [Hz]", self.freq2_hz, None),
            ("phase1_rad", "φ_in,1 [rad]", self.phase1_rad, None),
            ("phase2_rad", "φ_in,2 [rad]", self.phase2_rad, None),
            ("f_min_hz", "Untere Frequenz [Hz]", self.f_min_hz, None),
            ("f_max_hz", "Obere Frequenz [Hz]", self.f_max_hz, None),
            ("frequency_points", "Frequenzpunkte", self.frequency_points, None),
            ("harmonics", "Harmonische in THD-Auswertung", self.harmonics, None),
            ("x_min_mm", "Minimale Auslenkung [mm]", self.x_min_mm, None),
            ("x_max_mm", "Maximale Auslenkung [mm]", self.x_max_mm, None),
            ("points", "Stützstellen", self.points, None),
            ("signal_path", "Signaldatei", self.signal_path, "file"),
            ("normalise_external", "Auf $A_{in}$ normieren", self.normalise_external, "check"),
            ("record_internal", "Interne Zustände aufzeichnen", self.record_internal, "check"),
            ("reference_enabled", "Bernardini-Referenz anzeigen", self.reference_enabled, "check"),
            ("reference_path", "Referenzdatei", self.reference_path, "reference_file"),
        ]
        start_row = 5
        for offset, (key, label_text, variable, kind) in enumerate(params):
            row = start_row + offset
            label = ttk.Label(controls, text=label_text)
            label.grid(row=row, column=0, sticky="w", pady=2)
            if kind == "check":
                widget: tk.Widget = ttk.Checkbutton(controls, variable=variable)
            elif isinstance(kind, list):
                widget = ttk.Combobox(controls, textvariable=variable, values=kind, state="readonly")
                widget.bind("<<ComboboxSelected>>", lambda _event: self._update_frequency_entry())
            elif kind in {"file", "reference_file"}:
                holder = ttk.Frame(controls)
                entry = ttk.Entry(holder, textvariable=variable, width=20)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                command = self._choose_signal_file if kind == "file" else self._choose_reference_file
                ttk.Button(holder, text="…", width=3, command=command).pack(
                    side=tk.LEFT, padx=(4, 0)
                )
                widget = holder
            else:
                widget = ttk.Entry(controls, textvariable=variable)
            widget.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
            self._parameter_rows[key] = (label, widget)

        plot_row = start_row + len(params) + 1
        ttk.Separator(controls).grid(row=plot_row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(controls, text="Darstellung", font=("TkDefaultFont", 11, "bold")).grid(
            row=plot_row + 1, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )
        self.plot_box = ttk.Combobox(controls, textvariable=self.plot_kind, state="readonly")
        self.plot_box.grid(row=plot_row + 2, column=0, columnspan=3, sticky="ew")
        self.plot_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw_result())

        self.run_button = ttk.Button(controls, text="Untersuchung ausführen", command=self._start_run)
        self.run_button.grid(row=plot_row + 3, column=0, columnspan=3, sticky="ew", pady=(14, 5))
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.grid(row=plot_row + 4, column=0, columnspan=3, sticky="ew", pady=3)
        ttk.Label(controls, textvariable=self.status_text, wraplength=360).grid(
            row=plot_row + 5, column=0, columnspan=3, sticky="w", pady=(3, 8)
        )

        export_frame = ttk.Frame(controls)
        export_frame.grid(row=plot_row + 6, column=0, columnspan=3, sticky="ew")
        self.save_data_button = ttk.Button(
            export_frame, text="Daten speichern…", command=self._save_data, state=tk.DISABLED
        )
        self.save_data_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.save_plot_button = ttk.Button(
            export_frame, text="Plot speichern…", command=self._save_plot, state=tk.DISABLED
        )
        self.save_plot_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        controls.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        summary = ttk.LabelFrame(plot_area, text="Ergebnis", padding=8)
        summary.pack(side=tk.BOTTOM, fill=tk.X, pady=(7, 0))
        ttk.Label(summary, textvariable=self.summary_text, wraplength=800).pack(anchor="w")

        axis = self.figure.add_subplot(111)
        axis.text(0.5, 0.5, "Untersuchung konfigurieren und ausführen", ha="center", va="center")
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _show_rows(self, keys: set[str]) -> None:
        for key, (label, widget) in self._parameter_rows.items():
            if key in keys:
                label.grid()
                widget.grid()
            else:
                label.grid_remove()
                widget.grid_remove()

    def _experiment_changed(self) -> None:
        experiment = EXPERIMENTS[self.experiment_label.get()]
        if self.result is not None and self.result.experiment != experiment:
            self.result = None
            self.summary_text.set("Für die gewählte Untersuchung liegt noch kein Ergebnis vor.")
            self.save_data_button.configure(state=tk.DISABLED)
            self.save_plot_button.configure(state=tk.DISABLED)
        rows = {
            "small_signal_reference": {"reference_path"},
            "nonlinear_characteristics": {"x_min_mm", "x_max_mm", "points"},
            "stationary_sine": {
                "fs", "vrms", "input_level_mode", "duration_s", "settle_s", "frequency_mode",
                "freq_hz", "harmonics",
                "record_internal", "reference_enabled", "reference_path",
            },
            "thd_frequency_response": {
                "fs", "vrms", "input_level_mode", "f_min_hz", "f_max_hz",
                "frequency_points", "harmonics"
            },
            "two_tone": {
                "fs", "vrms", "input_level_mode", "duration_s", "settle_s", "freq1_hz", "freq2_hz",
                "phase1_rad", "phase2_rad", "record_internal", "reference_enabled", "reference_path",
            },
            "exponential_sweep": {
                "fs", "vrms", "input_level_mode", "duration_s", "f_min_hz", "f_max_hz",
                "record_internal"
            },
            "external_signal": {
                "fs", "vrms", "input_level_mode", "settle_s", "signal_path", "normalise_external",
                "record_internal",
                "reference_enabled", "reference_path",
            },
        }[experiment]
        self._show_rows(rows)
        options = PLOT_OPTIONS[experiment]
        self.plot_box.configure(values=options)
        self.plot_kind.set(options[0])
        self._update_frequency_entry()

    def _preset_changed(self) -> None:
        if EXPERIMENTS[self.experiment_label.get()] == "two_tone":
            defaults = wdf.get_paper_two_tone_defaults(self.preset_name.get())
            self.vrms.set(str(defaults["vrms"]))
            self.freq1_hz.set(str(defaults["freq1_hz"]))
            self.freq2_hz.set(str(defaults["freq2_hz"]))
            self.phase1_rad.set(str(defaults["phase1_rad"]))
            self.phase2_rad.set(str(defaults["phase2_rad"]))
        if EXPERIMENTS[self.experiment_label.get()] == "nonlinear_characteristics":
            x_min, x_max = wdf.get_fig5_displacement_defaults_mm(self.preset_name.get())
            self.x_min_mm.set(str(x_min))
            self.x_max_mm.set(str(x_max))

    def _update_frequency_entry(self) -> None:
        _label, widget = self._parameter_rows["freq_hz"]
        entry = widget
        if isinstance(entry, ttk.Entry):
            entry.configure(state=tk.NORMAL if self.frequency_mode.get() == "frei wählbar" else tk.DISABLED)

    def _choose_signal_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Eingangssignal auswählen",
            filetypes=[("Audiosignal", "*.wav *.npy"), ("WAV", "*.wav"), ("NumPy", "*.npy")],
        )
        if selected:
            self.signal_path.set(selected)

    def _choose_reference_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Bernardini-Referenzdaten auswählen",
            filetypes=[("Bernardini-TXT-Daten", "*.txt"), ("Textdatei", "*.txt")],
        )
        if not selected:
            return
        try:
            reference = comparison.load_reference_data(selected)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Ungültige Referenzdatei", str(exc), parent=self)
            return
        self.reference_path.set(selected)
        self.reference_enabled.set(True)
        self.preset_name.set(reference.preset_name)
        if reference.fs_hz is not None:
            self.fs.set(str(int(reference.fs_hz)))

        if reference.experiment in {"sinusoid", "two_tone", "music"}:
            maximum_level = 11.5 if reference.preset_name == "Spk-1" else 3.0
            self.vrms.set(str(maximum_level))
            self.input_level_mode.set(BERNARDINI_INPUT_LEVEL_MODE)
        if reference.experiment == "sinusoid":
            frequency = 28.75 if reference.preset_name == "Spk-1" else 81.4
            self.frequency_mode.set("frei wählbar")
            self.freq_hz.set(str(frequency))
        elif reference.experiment == "two_tone":
            defaults = wdf.get_paper_two_tone_defaults(reference.preset_name)
            self.freq1_hz.set(str(defaults["freq1_hz"]))
            self.freq2_hz.set(str(defaults["freq2_hz"]))
            self.phase1_rad.set(str(defaults["phase1_rad"]))
            self.phase2_rad.set(str(defaults["phase2_rad"]))

        compatible_plot = (
            "Zeitverlauf" if reference.domain == "time" else "Leistungsdichtespektrum"
        )
        if compatible_plot in tuple(self.plot_box.cget("values")):
            self.plot_kind.set(compatible_plot)

    def _collect_config(self) -> dict[str, Any]:
        return {
            "experiment": EXPERIMENTS[self.experiment_label.get()],
            "preset_name": self.preset_name.get(),
            "fs": float(self.fs.get()),
            "vrms": float(self.vrms.get()),
            "input_level_mode": self.input_level_mode.get(),
            "duration_s": float(self.duration_s.get()),
            "settle_s": float(self.settle_s.get()),
            "frequency_mode": self.frequency_mode.get(),
            "freq_hz": float(self.freq_hz.get()),
            "freq1_hz": float(self.freq1_hz.get()),
            "freq2_hz": float(self.freq2_hz.get()),
            "phase1_rad": float(self.phase1_rad.get()),
            "phase2_rad": float(self.phase2_rad.get()),
            "f_min_hz": float(self.f_min_hz.get()),
            "f_max_hz": float(self.f_max_hz.get()),
            "frequency_points": int(self.frequency_points.get()),
            "harmonics": int(self.harmonics.get()),
            "x_min_mm": float(self.x_min_mm.get()),
            "x_max_mm": float(self.x_max_mm.get()),
            "points": int(self.points.get()),
            "signal_path": self.signal_path.get(),
            "normalise_external": bool(self.normalise_external.get()),
            "record_internal": bool(self.record_internal.get()),
            "reference_enabled": bool(self.reference_enabled.get()),
            "reference_path": self.reference_path.get(),
        }

    def _start_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            config = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Ungültige Eingabe", str(exc), parent=self)
            return

        self.run_button.configure(state=tk.DISABLED)
        self.save_data_button.configure(state=tk.DISABLED)
        self.save_plot_button.configure(state=tk.DISABLED)
        self.status_text.set("Simulation läuft … Die Oberfläche bleibt währenddessen bedienbar.")
        self.progress.start(10)
        self._worker = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        self._worker.start()

    def _run_worker(self, config: dict[str, Any]) -> None:
        try:
            result = run_experiment(config)
            self._messages.put(("result", result))
        except Exception:
            self._messages.put(("error", traceback.format_exc()))

    def _poll_worker(self) -> None:
        try:
            kind, payload = self._messages.get_nowait()
        except queue.Empty:
            pass
        else:
            self.progress.stop()
            self.run_button.configure(state=tk.NORMAL)
            if kind == "result":
                self.result = payload
                notice = payload.data.get("reference_notice")
                self.summary_text.set(payload.summary if not notice else f"{payload.summary} {notice}")
                self.status_text.set("Untersuchung erfolgreich abgeschlossen.")
                self.save_data_button.configure(state=tk.NORMAL)
                self.save_plot_button.configure(state=tk.NORMAL)
                self._redraw_result()
            else:
                self.status_text.set("Untersuchung fehlgeschlagen.")
                details = str(payload)
                messagebox.showerror(
                    "Fehler bei der Simulation", details.splitlines()[-1] if details else "Unbekannter Fehler", parent=self
                )
        self.after(100, self._poll_worker)

    @staticmethod
    def _decimated(signal: np.ndarray, max_points: int = 12000) -> tuple[np.ndarray, int]:
        stride = max(1, int(np.ceil(len(signal) / max_points)))
        return np.asarray(signal)[::stride], stride

    def _redraw_result(self) -> None:
        if self.result is None:
            return
        result = self.result
        plot_kind = self.plot_kind.get()
        self.figure.clear()

        if result.experiment == "small_signal_reference":
            references = result.data["small_signal_references"]

            def plot_impedance(axis: Any) -> None:
                reference = references.get("electric_impedance_magnitude")
                if reference is None:
                    axis.text(0.5, 0.5, "Keine |Z_e|-Referenz geladen", ha="center", va="center")
                    axis.set_axis_off()
                    return
                axis.semilogx(reference.x, reference.values, label="Messung", color="tab:blue")
                axis.set_xlabel("Frequenz [Hz]")
                axis.set_ylabel(r"$|Z_e|$ [$\Omega$]")
                axis.set_title("Elektrische Eingangsimpedanz")
                axis.grid(True, which="both", linestyle=":", alpha=0.7)

            def plot_pressure_response(axis: Any) -> None:
                reference = references.get("output_pressure_magnitude")
                if reference is None:
                    axis.text(0.5, 0.5, "Keine |P_out|-Referenz geladen", ha="center", va="center")
                    axis.set_axis_off()
                    return
                axis.loglog(reference.x, reference.values, label="Messung", color="tab:blue")
                axis.set_xlabel("Frequenz [Hz]")
                axis.set_ylabel(r"$|P_{out}|$ [Pa]")
                axis.set_title("Ausgangsdruck-Frequenzgang")
                axis.grid(True, which="both", linestyle=":", alpha=0.7)

            if plot_kind == "Elektrische Eingangsimpedanz und Ausgangsdruck":
                axes = self.figure.subplots(2, 1, sharex=True)
                plot_impedance(axes[0])
                plot_pressure_response(axes[1])
                self.figure.suptitle(f"Kleinsignal-Messdaten – {result.preset_name}")
            elif plot_kind == "Elektrische Eingangsimpedanz":
                plot_impedance(self.figure.add_subplot(111))
            else:
                plot_pressure_response(self.figure.add_subplot(111))
            self.canvas.draw_idle()
            return

        if result.experiment == "nonlinear_characteristics":
            axis = self.figure.add_subplot(111)
            axis_right = axis.twinx()
            x = result.data["x_mm"]
            line1 = axis.plot(x, result.data["bl"], label=r"$Bl(x)$", color="tab:blue")
            line2 = axis.plot(
                x, result.data["kms_n_per_mm"], label=r"$K_{ms}(x)$", color="tab:green", linestyle="--"
            )
            line3 = axis_right.plot(x, result.data["le_mh"], label=r"$L_e(x)$", color="tab:orange")
            axis.set_title(f"Auslenkungsabhängige Modellparameter – {result.preset_name}")
            axis.set_xlabel(r"Membranauslenkung $x$ [mm]")
            axis.set_ylabel(r"$Bl$ [N/A], $K_{ms}$ [N/mm]")
            axis_right.set_ylabel(r"$L_e$ [mH]")
            lines = line1 + line2 + line3
            axis.legend(lines, [line.get_label() for line in lines], loc="best")
            axis.grid(True, linestyle=":", alpha=0.7)
            self.canvas.draw_idle()
            return

        if result.experiment == "thd_frequency_response":
            axis = self.figure.add_subplot(111)
            axis.semilogx(result.data["frequencies_hz"], result.data["thd_db"], marker="o", ms=4)
            axis.set_title(f"Frequenzabhängige harmonische Verzerrung – {result.preset_name}")
            axis.set_xlabel(r"Anregungsfrequenz $f_{in}$ [Hz]")
            axis.set_ylabel("THD [dB]")
            axis.grid(True, which="both", linestyle=":", alpha=0.7)
            self.canvas.draw_idle()
            return

        fs = float(result.fs)

        if plot_kind == "Leistungsdichtespektrum" and "reference_psd_comparison" in result.data:
            spectral = result.data["reference_psd_comparison"]
            reference = result.data["reference"]
            axes = self.figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            positive = spectral.frequency_hz > 0.0
            axes[0].semilogx(
                spectral.frequency_hz[positive],
                spectral.reference_db[positive],
                label="Bernardini-Messung",
                linewidth=1.0,
            )
            axes[0].semilogx(
                spectral.frequency_hz[positive],
                spectral.simulation_db[positive],
                label="Aktueller WDF-Solver",
                linewidth=1.0,
            )
            axes[0].set_ylabel(r"PSD von $P_{out}$ [dB/Hz]")
            axes[0].set_title(
                f"PSD-Referenzvergleich – Fig. {reference.figure}, {result.preset_name}"
            )
            axes[0].legend()
            axes[0].grid(True, which="both", linestyle=":", alpha=0.7)
            relevant = positive & spectral.relevant_mask
            axes[1].semilogx(
                spectral.frequency_hz[relevant],
                spectral.difference_db[relevant],
                color="tab:red",
                linewidth=0.9,
            )
            axes[1].axhline(0.0, color="black", linewidth=0.7)
            axes[1].set_xlabel("Frequenz [Hz]")
            axes[1].set_ylabel("Δ [dB]")
            axes[1].grid(True, which="both", linestyle=":", alpha=0.7)
            self.canvas.draw_idle()
            return

        if plot_kind == "Zeitverlauf":
            axis = self.figure.add_subplot(111)
            if "reference_time_comparison" in result.data:
                time_comparison = result.data["reference_time_comparison"]
                axis.plot(
                    time_comparison.time_s,
                    time_comparison.reference,
                    label=f"Bernardini-Messung ({time_comparison.descriptor.label})",
                    linewidth=1.0,
                )
                axis.plot(
                    time_comparison.time_s,
                    time_comparison.simulation,
                    label="Aktueller WDF-Solver (normiert und subsample-ausgerichtet)",
                    linewidth=1.0,
                    alpha=0.85,
                )
                axis.set_title(f"Zeitbereichs-Referenzvergleich – {result.preset_name}")
                axis.text(
                    0.99,
                    0.02,
                    (
                        f"NRMSE absolut = {time_comparison.nrmse:.5f} "
                        f"({100.0 * time_comparison.nrmse:.2f} %)\n"
                        f"NRMSE AC = {time_comparison.ac_nrmse:.5f} "
                        f"({100.0 * time_comparison.ac_nrmse:.2f} %)\n"
                        f"Zeitversatz = {time_comparison.alignment_start_samples:.3f} Samples\n"
                        f"N = {time_comparison.comparison_samples}"
                    ),
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
                )
            else:
                output, output_stride = self._decimated(result.data["output"])
                input_signal, input_stride = self._decimated(result.data["input"])
                t_out = np.arange(len(output)) * output_stride / fs
                t_in = np.arange(len(input_signal)) * input_stride / fs
                axis.plot(t_in, input_signal, label=r"$V_{in}(t)$", linewidth=0.9)
                axis.plot(t_out, output, label=r"$P_{out}(t)$", linewidth=0.9, alpha=0.85)
                axis.set_title(f"Ein- und Ausgangssignal – {result.preset_name}")
            axis.set_xlabel("Zeit [s]")
            axis.set_ylabel(r"$P_{out}$ [Pa]" if "reference_time_comparison" in result.data else "Amplitude")
            axis.legend()
            axis.grid(True, linestyle=":", alpha=0.7)
        elif plot_kind == "Harmonische Komponenten":
            axis = self.figure.add_subplot(111)
            items = result.data["harmonic_data"]
            harmonics = [item["harmonic"] for item in items]
            levels = [item["relative_db"] for item in items]
            axis.bar(harmonics, levels, color="tab:blue")
            axis.set_title(
                f"Harmonische Komponenten bei $f_{{in}}$ = {result.data['freq_hz']:.3f} Hz"
            )
            axis.set_xlabel("Ordnung der Harmonischen")
            axis.set_ylabel("Pegel relativ zum Grundton [dB]")
            axis.set_xticks(harmonics)
            axis.grid(True, axis="y", linestyle=":", alpha=0.7)
        elif plot_kind in {"Membranauslenkung", "Membrangeschwindigkeit", "Schwingspulenstrom"}:
            axis = self.figure.add_subplot(111)
            trace_definitions = {
                "Membranauslenkung": ("displacement", 1e3, r"Membranauslenkung $x(t)$ [mm]"),
                "Membrangeschwindigkeit": (
                    "diaphragm_velocity", 1.0, r"Membrangeschwindigkeit $I_{ms}(t)$ [m/s]"
                ),
                "Schwingspulenstrom": ("voice_coil_current", 1.0, r"Schwingspulenstrom $I_e(t)$ [A]"),
            }
            trace_key, scale, ylabel = trace_definitions[plot_kind]
            if trace_key not in result.data:
                axis.text(
                    0.5,
                    0.5,
                    "Diese Größe wurde nicht aufgezeichnet.\n"
                    "„Interne Zustände aufzeichnen“ aktivieren und die Untersuchung erneut ausführen.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            else:
                trace, stride = self._decimated(result.data[trace_key])
                time_s = np.arange(len(trace)) * stride / fs
                axis.plot(time_s, scale * trace, linewidth=0.9)
                axis.set_title(f"{plot_kind} – {result.preset_name}")
                axis.set_xlabel("Zeit [s]")
                axis.set_ylabel(ylabel)
                axis.grid(True, linestyle=":", alpha=0.7)
        else:
            axis = self.figure.add_subplot(111)
            analysis = result.data["analysis"]
            if plot_kind == "Leistungsdichtespektrum":
                if result.experiment == "two_tone":
                    freqs, levels = wdf.compute_highres_psd_db(analysis, fs)
                else:
                    freqs, levels = wdf.compute_psd_db(analysis, fs)
                ylabel = r"PSD von $P_{out}$ [dB/Hz]"
                title = "Leistungsdichtespektrum des Ausgangsdrucks"
            else:
                freqs, levels = wdf.compute_power_spectrum_db(analysis, fs)
                ylabel = r"Leistung von $P_{out}$ [dB]"
                title = "Leistungsspektrum des Ausgangsdrucks"
            mask = (freqs > 0.0) & (freqs <= fs / 2.0)
            axis.semilogx(freqs[mask], levels[mask], linewidth=0.9)
            axis.set_title(f"{title} – {result.preset_name}")
            axis.set_xlabel("Frequenz [Hz]")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", linestyle=":", alpha=0.7)
        self.canvas.draw_idle()

    def _save_data(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(
            title="Simulationsergebnis speichern",
            defaultextension=".npz",
            filetypes=[("Komprimiertes NumPy-Archiv", "*.npz")],
        )
        if not path:
            return
        arrays = {
            key: value
            for key, value in self.result.data.items()
            if isinstance(value, np.ndarray) and value.dtype != object
        }
        arrays["preset_name"] = np.asarray(self.result.preset_name)
        arrays["experiment"] = np.asarray(self.result.experiment)
        arrays["fs_hz"] = np.asarray(np.nan if self.result.fs is None else self.result.fs)
        np.savez_compressed(path, **arrays)

        metadata: dict[str, Any] = {
            "experiment": self.result.experiment,
            "preset_name": self.result.preset_name,
            "fs_hz": self.result.fs,
            "summary": self.result.summary,
            "array_keys": sorted(arrays),
        }
        for key, value in self.result.data.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
            elif isinstance(value, np.generic):
                metadata[key] = value.item()
            elif key in {"harmonic_data", "reference_metadata"}:
                metadata[key] = value
        metadata_path = Path(path).with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_text.set(f"Ergebnis und Metadaten gespeichert: {path}")

    def _save_plot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Darstellung speichern",
            defaultextension=".png",
            filetypes=[("PNG-Grafik", "*.png"), ("PDF-Grafik", "*.pdf"), ("SVG-Grafik", "*.svg")],
        )
        if path:
            self.figure.savefig(path, dpi=180, bbox_inches="tight")
            self.status_text.set(f"Plot gespeichert: {path}")


def main() -> None:
    app = WDFSimulatorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
