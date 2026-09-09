"""Import und Vergleich der von Bernardini et al. bereitgestellten Daten.

Das Modul ist absichtlich unabhaengig von Tkinter und vom WDF-Solver. Es
beschreibt Referenzdaten, Spektralkonfigurationen und Einheiten explizit, damit
Dateiimport, Simulation und Darstellung nicht miteinander vermischt werden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, correlation_lags, welch


SPEAKER_TO_PRESET = {
    "FSF122.02-8": "Spk-1",
    "FSN020.71-4": "Spk-2",
}

PRESET_TO_SPEAKER = {value: key for key, value in SPEAKER_TO_PRESET.items()}


@dataclass(frozen=True)
class PSDConfig:
    """Vollstaendige Konfiguration einer Welch-PSD-Schaetzung."""

    fs_hz: float
    window: str = "hamming"
    segment_length: int = 19200
    overlap: int = 9600
    nfft: int = 19200
    detrend: bool = False
    scaling: str = "density"
    confirmed_properties: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    @property
    def frequency_spacing_hz(self) -> float:
        return float(self.fs_hz) / int(self.nfft)


@dataclass(frozen=True)
class SeriesDescriptor:
    """Physikalische Parameter einer Spalte in einer Mehrfachdatei."""

    column: int
    label: str
    vrms: float | None = None
    frequency_hz: float | None = None


@dataclass
class ReferenceData:
    """Normalisierte Beschreibung einer Bernardini-Referenzdatei."""

    path: Path
    figure: str
    preset_name: str
    speaker_model: str
    source_type: str
    experiment: str
    quantity: str
    domain: str
    x: np.ndarray
    values: np.ndarray
    x_unit: str
    value_unit: str
    value_scale: str
    series: list[SeriesDescriptor] = field(default_factory=list)
    fs_hz: float | None = None
    psd_config: PSDConfig | None = None
    expected_vrms: float | None = None
    expected_frequencies_hz: tuple[float, ...] = ()
    expected_phases_rad: tuple[float, ...] = ()
    applied_value_scale_factor: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def frequency_spacing_hz(self) -> float | None:
        if self.domain != "frequency" or len(self.x) < 2:
            return None
        differences = np.diff(self.x)
        if np.allclose(differences, differences[0], rtol=1e-8, atol=1e-10):
            return float(differences[0])
        return None


@dataclass(frozen=True)
class SpectrumComparison:
    """Zwei PSD-Kurven auf demselben Frequenzraster."""

    frequency_hz: np.ndarray
    reference_linear: np.ndarray
    simulation_linear: np.ndarray
    reference_db: np.ndarray
    simulation_db: np.ndarray
    difference_db: np.ndarray
    relevant_mask: np.ndarray


@dataclass(frozen=True)
class TimeSeriesComparison:
    """Referenz, ausgerichtete Simulation und zeitbezogene Fehlermaße.

    ``nrmse`` ist als absoluter relativer L2-Fehler definiert:
    ||simulation - reference||_2 / ||reference||_2. Vor der Berechnung wird
    ausschließlich eine Zeitverschiebung korrigiert; eine Pegelanpassung findet
    nicht statt. ``ac_nrmse`` wird nach separater Mittelwertentfernung berechnet
    und bewertet damit die zeitveränderliche Druckkomponente ohne DC-Offset.
    """

    time_s: np.ndarray
    reference: np.ndarray
    simulation: np.ndarray
    lag_samples: int
    descriptor: SeriesDescriptor
    rmse_pa: float
    reference_rms_pa: float
    nrmse: float
    ac_rmse_pa: float
    ac_reference_rms_pa: float
    ac_nrmse: float
    fractional_offset_samples: float
    alignment_correlation: float

    @property
    def alignment_start_samples(self) -> float:
        return float(self.lag_samples) + float(self.fractional_offset_samples)

    @property
    def ac_level_difference_db(self) -> float:
        """RMS-Pegeldifferenz der mittelwertfreien Drucksignale in dB.

        Positive Werte kennzeichnen einen hoeheren Simulationspegel, negative
        Werte einen niedrigeren Simulationspegel gegenueber der Messreferenz.
        """
        simulation_ac = self.simulation - np.mean(self.simulation)
        simulation_ac_rms = float(np.sqrt(np.mean(simulation_ac * simulation_ac)))
        if simulation_ac_rms <= np.finfo(float).tiny:
            return float("-inf")
        return float(20.0 * np.log10(simulation_ac_rms / self.ac_reference_rms_pa))

    @property
    def comparison_samples(self) -> int:
        return int(len(self.reference))

    @property
    def duration_s(self) -> float:
        if len(self.time_s) < 2:
            return 0.0
        return float(len(self.time_s) * (self.time_s[1] - self.time_s[0]))


def _speaker_from_filename(filename: str) -> tuple[str, str]:
    for speaker_model, preset_name in SPEAKER_TO_PRESET.items():
        if filename.startswith(speaker_model):
            return speaker_model, preset_name
    raise ValueError(f"Unbekanntes Lautsprechermodell im Dateinamen: {filename}")


def _source_from_filename(filename: str) -> str:
    if filename.endswith("_meas.txt"):
        return "measurement"
    if filename.endswith("_wdf.txt"):
        return "bernardini_wdf"
    if filename.endswith("_spice.txt"):
        return "spice"
    return "unknown"


def _figure_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent.startswith("fig"):
        return parent[3:]
    return "unknown"


def _time_series_descriptors(experiment: str, preset_name: str, columns: int) -> list[SeriesDescriptor]:
    levels = [4.0, 7.5, 11.5] if preset_name == "Spk-1" else [1.0, 2.0, 3.0]

    if experiment == "sinusoid":
        frequencies = [28.75, 57.5, 115.0] if preset_name == "Spk-1" else [81.4, 162.8, 325.6]
        descriptors = []
        column = 0
        for frequency_hz in frequencies:
            for vrms in levels:
                descriptors.append(
                    SeriesDescriptor(
                        column=column,
                        label=f"{vrms:g} V RMS, {frequency_hz:g} Hz",
                        vrms=vrms,
                        frequency_hz=frequency_hz,
                    )
                )
                column += 1
        if columns != len(descriptors):
            raise ValueError(
                f"Für Sinus-Referenzdaten wurden {len(descriptors)} Spalten erwartet, erhalten: {columns}."
            )
        return descriptors

    if experiment in {"two_tone", "music"}:
        if columns != len(levels):
            raise ValueError(
                f"Für {experiment} wurden {len(levels)} Spalten erwartet, erhalten: {columns}."
            )
        return [
            SeriesDescriptor(column=index, label=f"{vrms:g} V RMS", vrms=vrms)
            for index, vrms in enumerate(levels)
        ]

    return [SeriesDescriptor(column=index, label=f"Spalte {index + 1}") for index in range(columns)]


def _validate_linear_file_columns(raw: np.ndarray) -> None:
    """Akzeptiert Frequenz/Betrag und die in den Messdateien leere dritte Spalte."""
    if raw.shape[1] == 2:
        return
    if raw.shape[1] == 3 and np.all(raw[:, 2] == 0.0):
        return
    raise ValueError(
        "Lineare Referenzdateien müssen Frequenz und Betrag enthalten; eine dritte Spalte "
        "ist nur zulässig, wenn sie vollständig null ist."
    )


def _expected_psd_excitation(
    experiment: str,
    preset_name: str,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Liefert die in Fig. 6(c/d), 7(c/d) und 8(c/d) dargestellte Anregung."""
    maximum_level = 11.5 if preset_name == "Spk-1" else 3.0
    if experiment == "sinusoid":
        frequency = 28.75 if preset_name == "Spk-1" else 81.4
        return maximum_level, (frequency,), ()
    if experiment == "two_tone":
        frequencies = (28.75, 121.78) if preset_name == "Spk-1" else (81.4, 344.79)
        return maximum_level, frequencies, (0.8472, 2.2179)
    return maximum_level, (), ()


def infer_psd_config(reference: ReferenceData, fs_hz: float = 96000.0) -> PSDConfig:
    """Leitet sichere PSD-Parameter ab und markiert alle Annahmen explizit."""
    spacing = reference.frequency_spacing_hz
    if spacing is None or spacing <= 0.0:
        raise ValueError("Aus der Referenzdatei kann kein gleichmaessiges PSD-Raster abgeleitet werden.")
    nfft = int(round(float(fs_hz) / spacing))
    if nfft < 2 or not np.isclose(float(fs_hz) / nfft, spacing, rtol=1e-8, atol=1e-10):
        raise ValueError("Referenzraster und Abtastrate ergeben keine ganzzahlige FFT-Laenge.")
    expected_points = nfft // 2 + 1
    if len(reference.x) != expected_points:
        raise ValueError(
            f"Das Referenzraster besitzt {len(reference.x)} Punkte; für eine einseitige FFT "
            f"der Länge {nfft} werden {expected_points} erwartet."
        )
    if not np.isclose(reference.x[0], 0.0) or not np.isclose(reference.x[-1], fs_hz / 2.0):
        raise ValueError("Das PSD-Referenzraster reicht nicht exakt von DC bis Nyquist.")
    return PSDConfig(
        fs_hz=float(fs_hz),
        window="hamming",
        segment_length=nfft,
        overlap=nfft // 2,
        nfft=nfft,
        detrend=False,
        scaling="density",
        confirmed_properties=(
            "Hamming-Fenster und Segmentlaenge gleich FFT-Laenge stimmen mit dem Leakage-Muster "
            "mehrerer Grundtoene in Fig. 6(c), 7(c) und 7(d) ueberein",
        ),
        assumptions=(
            "50 % Segmentueberlappung angenommen",
            "Kein Detrending angenommen",
        ),
    )


def load_reference_data(path: str | Path) -> ReferenceData:
    """Laedt und klassifiziert eine Bernardini-TXT-Datei anhand ihres Namens."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Referenzdatei nicht gefunden: {source_path}")
    raw = np.loadtxt(source_path, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("Referenzdateien muessen mindestens zwei numerische Spalten enthalten.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Die Referenzdatei enthaelt nicht-endliche Werte.")

    filename = source_path.name
    speaker_model, preset_name = _speaker_from_filename(filename)
    figure = _figure_from_path(source_path)
    source_type = _source_from_filename(filename)

    if "_linear_Ze_" in filename:
        _validate_linear_file_columns(raw)
        if not np.all(np.diff(raw[:, 0]) > 0.0):
            raise ValueError("Die Frequenzachse ist nicht streng monoton steigend.")
        return ReferenceData(
            source_path,
            figure,
            preset_name,
            speaker_model,
            source_type,
            "small_signal",
            "electric_impedance_magnitude",
            "frequency",
            raw[:, 0],
            raw[:, 1],
            "Hz",
            "ohm",
            "linear",
            notes=["Eine eventuell vorhandene dritte Nullspalte wird ignoriert."],
        )

    if "_linear_P_" in filename:
        _validate_linear_file_columns(raw)
        if not np.all(np.diff(raw[:, 0]) > 0.0):
            raise ValueError("Die Frequenzachse ist nicht streng monoton steigend.")
        return ReferenceData(
            source_path,
            figure,
            preset_name,
            speaker_model,
            source_type,
            "small_signal",
            "output_pressure_magnitude",
            "frequency",
            raw[:, 0],
            raw[:, 1],
            "Hz",
            "Pa",
            "linear",
            notes=["Eine eventuell vorhandene dritte Nullspalte wird ignoriert."],
        )

    is_psd = "_P_psd_" in filename
    if "_sin_" in filename:
        experiment = "sinusoid"
    elif "_multi_" in filename:
        experiment = "two_tone"
    elif "_audio_" in filename:
        experiment = "music"
    else:
        raise ValueError(f"Nicht unterstuetzter Bernardini-Dateityp: {filename}")

    if is_psd:
        if raw.shape[1] != 2:
            raise ValueError("PSD-Referenzdateien muessen genau zwei Spalten enthalten.")
        if not np.all(np.diff(raw[:, 0]) > 0.0):
            raise ValueError("Die Frequenzachse ist nicht streng monoton steigend.")
        if np.any(raw[:, 1] < 0.0):
            raise ValueError("Lineare PSD-Werte duerfen nicht negativ sein.")
        expected_vrms, expected_frequencies_hz, expected_phases_rad = _expected_psd_excitation(
            experiment, preset_name
        )
        values = np.asarray(raw[:, 1], dtype=float)
        scale_factor = 1.0
        notes: list[str] = []
        if figure in {"6c", "6d"}:
            # Beide bereitgestellten Fig.-6-Dateien sind gegen ihre zugehoerigen
            # Zeitreihen um nahezu exakt 10^6 skaliert. Nach dieser Korrektur
            # erfuellt das PSD-Integral Parseval bis auf ca. 1 %.
            scale_factor = 1e-6
            values = values * scale_factor
            notes.append(
                "Fig.-6-PSD mit Faktor 1e-6 auf Pa^2/Hz normiert; bestätigt durch "
                "Parseval-Konsistenz mit den bereitgestellten Fig.-6-Zeitreihen beider Lautsprecher."
            )
        reference = ReferenceData(
            source_path,
            figure,
            preset_name,
            speaker_model,
            source_type,
            experiment,
            "output_pressure_psd",
            "frequency",
            raw[:, 0],
            values,
            "Hz",
            "Pa^2/Hz",
            "linear",
            fs_hz=96000.0,
            expected_vrms=expected_vrms,
            expected_frequencies_hz=expected_frequencies_hz,
            expected_phases_rad=expected_phases_rad,
            applied_value_scale_factor=scale_factor,
            notes=notes,
        )
        reference.psd_config = infer_psd_config(reference, fs_hz=96000.0)
        return reference

    values = raw
    descriptors = _time_series_descriptors(experiment, preset_name, values.shape[1])
    fs_hz = 96000.0
    time_s = np.arange(values.shape[0], dtype=float) / fs_hz
    expected_frequencies_hz: tuple[float, ...] = ()
    expected_phases_rad: tuple[float, ...] = ()
    if experiment == "two_tone":
        _, expected_frequencies_hz, expected_phases_rad = _expected_psd_excitation(
            experiment, preset_name
        )
    return ReferenceData(
        source_path,
        figure,
        preset_name,
        speaker_model,
        source_type,
        experiment,
        "output_pressure",
        "time",
        time_s,
        values,
        "s",
        "Pa",
        "linear",
        series=descriptors,
        fs_hz=fs_hz,
        expected_frequencies_hz=expected_frequencies_hz,
        expected_phases_rad=expected_phases_rad,
        notes=["Die Zeitachse ist aus Fs = 96 kHz abgeleitet und nicht in der TXT-Datei enthalten."],
    )


def find_companion_reference(reference: ReferenceData, quantity: str) -> Path | None:
    """Findet z.B. die zu einer P-Datei gehoerende Ze-Datei im selben Ordner."""
    filename = reference.path.name
    if quantity == "electric_impedance_magnitude":
        candidate_name = filename.replace("_linear_P_", "_linear_Ze_")
    elif quantity == "output_pressure_magnitude":
        candidate_name = filename.replace("_linear_Ze_", "_linear_P_")
    else:
        return None
    candidate = reference.path.with_name(candidate_name)
    return candidate if candidate.exists() else None


def find_time_domain_companion(reference: ReferenceData) -> Path | None:
    """Findet zu einer PSD aus Fig. 6/7/8 die bereitgestellte Zeitmessung."""
    if reference.quantity != "output_pressure_psd":
        return None
    figure_pairs = {
        "6c": "6a",
        "6d": "6b",
        "7c": "7a",
        "7d": "7b",
        "8c": "8a",
        "8d": "8b",
    }
    time_figure = figure_pairs.get(reference.figure)
    if time_figure is None:
        return None
    filename = reference.path.name.replace("_P_psd_", "_P_")
    candidate = reference.path.parent.parent / f"fig{time_figure}" / filename
    return candidate if candidate.exists() else None


def select_series(
    reference: ReferenceData,
    vrms: float,
    frequency_hz: float | None = None,
) -> tuple[np.ndarray, SeriesDescriptor]:
    """Waehlt nur eine wirklich passende Spalte einer Zeitreferenz.

    Ein stilles Ausweichen auf die naechste Frequenz oder den naechsten Pegel
    waere fuer einen Validierungsvergleich irrefuehrend und ist daher verboten.
    """
    if reference.domain != "time" or reference.values.ndim != 2:
        raise ValueError("Die Referenz enthaelt keine waehlbaren Zeitreihen.")

    candidates = reference.series
    if frequency_hz is not None:
        candidates = [
            item
            for item in candidates
            if item.frequency_hz is not None
            and np.isclose(float(item.frequency_hz), frequency_hz, rtol=1e-7, atol=1e-7)
        ]
    if not candidates:
        available = sorted(
            {float(item.frequency_hz) for item in reference.series if item.frequency_hz is not None}
        )
        raise ValueError(
            f"Keine Referenzspalte für {frequency_hz:g} Hz vorhanden. Verfügbar: {available}."
        )
    level_matches = [
        item
        for item in candidates
        if item.vrms is not None and np.isclose(float(item.vrms), vrms, rtol=1e-7, atol=1e-7)
    ]
    if not level_matches:
        available = sorted({float(item.vrms) for item in candidates if item.vrms is not None})
        raise ValueError(
            f"Keine Referenzspalte für {vrms:g} V RMS vorhanden. Verfügbar: {available}."
        )
    descriptor = level_matches[0]
    return np.asarray(reference.values[:, descriptor.column], dtype=float), descriptor


def validate_reference_excitation(
    reference: ReferenceData,
    vrms: float,
    frequencies_hz: tuple[float, ...] = (),
    phases_rad: tuple[float, ...] = (),
) -> None:
    """Verhindert PSD-Vergleiche mit einer anderen Anregung als der Referenz."""
    if reference.expected_vrms is not None and not np.isclose(
        float(vrms), reference.expected_vrms, rtol=1e-7, atol=1e-7
    ):
        raise ValueError(
            f"Die PSD-Referenz gehört zu {reference.expected_vrms:g} V RMS, "
            f"die Simulation verwendet {vrms:g} V RMS."
        )
    expected = reference.expected_frequencies_hz
    if expected:
        actual = tuple(float(value) for value in frequencies_hz)
        if len(actual) != len(expected) or not np.allclose(actual, expected, rtol=1e-7, atol=1e-7):
            raise ValueError(
                f"Die PSD-Referenz gehört zu den Frequenzen {expected}, "
                f"die Simulation verwendet {actual}."
            )
    expected_phases = reference.expected_phases_rad
    if expected_phases:
        actual_phases = tuple(float(value) for value in phases_rad)
        if len(actual_phases) != len(expected_phases) or not np.allclose(
            actual_phases, expected_phases, rtol=1e-7, atol=1e-7
        ):
            raise ValueError(
                f"Die PSD-Referenz gehört zu den Phasen {expected_phases} rad, "
                f"die Simulation verwendet {actual_phases} rad."
            )


def compute_linear_psd(signal: np.ndarray, config: PSDConfig) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet eine lineare, einseitige Welch-PSD mit dokumentierter Konfiguration."""
    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Fuer die PSD wird ein nicht-leeres eindimensionales Signal benoetigt.")
    if len(samples) < config.segment_length:
        raise ValueError(
            f"Das Analysesignal besitzt {len(samples)} Samples, benoetigt werden mindestens "
            f"{config.segment_length}."
        )
    if config.nfft < config.segment_length:
        raise ValueError("Die FFT-Laenge darf nicht kleiner als die Segmentlaenge sein.")
    if not 0 <= config.overlap < config.segment_length:
        raise ValueError("Die Ueberlappung muss zwischen null und Segmentlaenge minus eins liegen.")
    frequencies, psd = welch(
        samples,
        fs=config.fs_hz,
        window=config.window,
        nperseg=config.segment_length,
        noverlap=config.overlap,
        nfft=config.nfft,
        scaling=config.scaling,
        detrend=config.detrend,
        return_onesided=True,
    )
    return frequencies, psd


def power_density_to_db(values: np.ndarray, floor: float = 1e-24) -> np.ndarray:
    """Konvertiert lineare Leistungsdichte nach dB relativ zu einer Einheit pro Hz."""
    return 10.0 * np.log10(np.maximum(np.asarray(values, dtype=float), float(floor)))


def compare_psd(
    simulation_signal: np.ndarray,
    simulation_fs_hz: float,
    reference: ReferenceData,
    relevant_dynamic_range_db: float = 80.0,
) -> SpectrumComparison:
    """Wertet eine Simulation exakt auf dem Frequenzraster einer PSD-Referenz aus."""
    if reference.quantity != "output_pressure_psd" or reference.psd_config is None:
        raise ValueError("Die Datei ist keine unterstuetzte PSD-Referenz.")
    if not np.isclose(simulation_fs_hz, reference.psd_config.fs_hz):
        raise ValueError(
            f"Abtastraten stimmen nicht ueberein: Simulation {simulation_fs_hz:g} Hz, "
            f"Referenz {reference.psd_config.fs_hz:g} Hz."
        )
    frequencies, simulation_psd = compute_linear_psd(simulation_signal, reference.psd_config)
    if not np.allclose(frequencies, reference.x, rtol=1e-9, atol=1e-9):
        raise ValueError("Simulation und Referenz besitzen trotz Konfiguration verschiedene Frequenzraster.")
    reference_db = power_density_to_db(reference.values)
    simulation_db = power_density_to_db(simulation_psd)
    relevant_mask = reference_db >= np.max(reference_db) - abs(float(relevant_dynamic_range_db))
    return SpectrumComparison(
        frequencies,
        np.asarray(reference.values, dtype=float),
        simulation_psd,
        reference_db,
        simulation_db,
        simulation_db - reference_db,
        relevant_mask,
    )


def align_time_series(
    simulation_signal: np.ndarray,
    simulation_fs_hz: float,
    reference: ReferenceData,
    vrms: float,
    frequency_hz: float | None = None,
    simulation_start_s: float = 0.0,
) -> TimeSeriesComparison:
    """Waehlt eine Referenzspalte und richtet sie pegelneutral im Zeitbereich aus.

    Der ganzzahlige Lag maximiert die normierte Kreuzkorrelation und ist daher
    nicht vom Energiepegel eines Kandidatenfensters abhaengig. Anschliessend
    wird lokal ein Subsample-Offset bestimmt, der den mittelwertfreien L2-Fehler
    minimiert. Es findet dabei keine multiplikative Pegelanpassung statt.
    """
    if reference.domain != "time" or reference.fs_hz is None:
        raise ValueError("Die Datei ist keine Zeitbereichsreferenz.")
    if not np.isclose(simulation_fs_hz, reference.fs_hz):
        raise ValueError(
            f"Abtastraten stimmen nicht ueberein: Simulation {simulation_fs_hz:g} Hz, "
            f"Referenz {reference.fs_hz:g} Hz."
        )
    measured, descriptor = select_series(reference, vrms=vrms, frequency_hz=frequency_hz)
    simulation = np.asarray(simulation_signal, dtype=float)
    if simulation.ndim != 1 or not np.all(np.isfinite(simulation)):
        raise ValueError("Das Simulationssignal muss eindimensional und endlich sein.")
    start = max(0, int(round(float(simulation_start_s) * simulation_fs_hz)))
    search = simulation[start:]
    if len(search) < len(measured):
        raise ValueError(
            f"Der Simulationsausschnitt ist mit {len(search)} Samples kuerzer als die "
            f"Referenz mit {len(measured)} Samples."
        )

    reference_centered = measured - np.mean(measured)
    reference_energy = float(np.dot(reference_centered, reference_centered))
    if reference_energy <= np.finfo(float).tiny:
        raise ValueError(
            "Die mittelwertfreie Messreferenz besitzt keine Energie; ein AC-NRMSE ist nicht definiert."
        )

    # Da die Referenz Summe null besitzt, entspricht der Zaehler bereits dem
    # Skalarprodukt der beiden lokal mittelwertfreien Signale. Die Energie jedes
    # Kandidatenfensters wird dagegen separat und effizient ueber Laufsummen
    # bestimmt. So bevorzugt die Ausrichtung keine lediglich lauteren Fenster.
    correlation = correlate(search, reference_centered, mode="valid", method="fft")
    window_length = len(measured)
    cumulative = np.concatenate(([0.0], np.cumsum(search, dtype=float)))
    cumulative_squared = np.concatenate(([0.0], np.cumsum(search * search, dtype=float)))
    window_sum = cumulative[window_length:] - cumulative[:-window_length]
    window_squared_sum = (
        cumulative_squared[window_length:] - cumulative_squared[:-window_length]
    )
    window_energy = window_squared_sum - window_sum * window_sum / window_length
    window_energy = np.maximum(window_energy, 0.0)
    valid_windows = window_energy > np.finfo(float).tiny
    if not np.any(valid_windows):
        raise ValueError("Der Simulationssuchbereich besitzt keine zeitveraenderliche Energie.")

    normalized_correlation = np.full_like(correlation, -np.inf, dtype=float)
    normalized_correlation[valid_windows] = correlation[valid_windows] / np.sqrt(
        reference_energy * window_energy[valid_windows]
    )
    finite_scores = np.isfinite(normalized_correlation)
    normalized_correlation[finite_scores] = np.clip(
        normalized_correlation[finite_scores], -1.0, 1.0
    )
    lags = correlation_lags(len(search), len(reference_centered), mode="valid")
    best_index = int(np.argmax(normalized_correlation))
    lag = int(lags[best_index])
    integer_start = start + lag

    # Lokale kubische Interpolation erlaubt eine Subsample-Ausrichtung, ohne
    # das Simulationssignal global neu abzutasten oder seinen Pegel zu skalieren.
    sample_offsets = np.arange(window_length, dtype=float)
    lower_offset = max(-1.0, -float(integer_start))
    upper_offset = min(
        1.0,
        float(len(simulation) - 1 - (integer_start + window_length - 1)),
    )
    interpolation_start = max(0, integer_start - 3)
    interpolation_stop = min(len(simulation), integer_start + window_length + 3)
    interpolation_indices = np.arange(interpolation_start, interpolation_stop, dtype=float)
    interpolator = CubicSpline(
        interpolation_indices,
        simulation[interpolation_start:interpolation_stop],
        extrapolate=False,
    )

    def fractional_ac_nrmse(offset_samples: float) -> float:
        candidate = interpolator(integer_start + sample_offsets + offset_samples)
        candidate_centered = candidate - np.mean(candidate)
        difference = candidate_centered - reference_centered
        return float(np.sqrt(np.dot(difference, difference) / reference_energy))

    fractional_offset = 0.0
    if upper_offset - lower_offset > 1e-12:
        optimum = minimize_scalar(
            fractional_ac_nrmse,
            bounds=(lower_offset, upper_offset),
            method="bounded",
            options={"xatol": 1e-7},
        )
        if optimum.success and np.isfinite(optimum.fun):
            fractional_offset = float(optimum.x)

    aligned = np.asarray(
        interpolator(integer_start + sample_offsets + fractional_offset), dtype=float
    )
    time_s = np.arange(len(measured), dtype=float) / simulation_fs_hz
    error = aligned - measured
    rmse_pa = float(np.sqrt(np.mean(error * error)))
    reference_rms_pa = float(np.sqrt(np.mean(measured * measured)))
    if reference_rms_pa <= np.finfo(float).tiny:
        raise ValueError("Der RMS-Wert der Messreferenz ist null; ein NRMSE ist nicht definiert.")
    nrmse = rmse_pa / reference_rms_pa
    aligned_centered = aligned - np.mean(aligned)
    ac_error = aligned_centered - reference_centered
    ac_rmse_pa = float(np.sqrt(np.mean(ac_error * ac_error)))
    ac_reference_rms_pa = float(np.sqrt(np.mean(reference_centered * reference_centered)))
    ac_nrmse = ac_rmse_pa / ac_reference_rms_pa
    aligned_energy = float(np.dot(aligned_centered, aligned_centered))
    if aligned_energy <= np.finfo(float).tiny:
        alignment_correlation = 0.0
    else:
        alignment_correlation = float(
            np.dot(aligned_centered, reference_centered)
            / np.sqrt(aligned_energy * reference_energy)
        )
    return TimeSeriesComparison(
        time_s,
        measured,
        aligned,
        integer_start,
        descriptor,
        rmse_pa,
        reference_rms_pa,
        float(nrmse),
        ac_rmse_pa,
        ac_reference_rms_pa,
        float(ac_nrmse),
        fractional_offset,
        alignment_correlation,
    )


def describe_reference(reference: ReferenceData) -> dict[str, Any]:
    """Liefert serialisierbare Metadaten fuer GUI und Ergebnisarchive."""
    result: dict[str, Any] = {
        "path": str(reference.path),
        "figure": reference.figure,
        "preset_name": reference.preset_name,
        "speaker_model": reference.speaker_model,
        "source_type": reference.source_type,
        "experiment": reference.experiment,
        "quantity": reference.quantity,
        "domain": reference.domain,
        "x_unit": reference.x_unit,
        "value_unit": reference.value_unit,
        "value_scale": reference.value_scale,
        "applied_value_scale_factor": reference.applied_value_scale_factor,
        "fs_hz": reference.fs_hz,
        "expected_vrms": reference.expected_vrms,
        "expected_frequencies_hz": list(reference.expected_frequencies_hz),
        "expected_phases_rad": list(reference.expected_phases_rad),
        "notes": list(reference.notes),
    }
    if reference.psd_config is not None:
        result["psd_config"] = {
            "window": reference.psd_config.window,
            "segment_length": reference.psd_config.segment_length,
            "overlap": reference.psd_config.overlap,
            "nfft": reference.psd_config.nfft,
            "detrend": reference.psd_config.detrend,
            "scaling": reference.psd_config.scaling,
            "confirmed_properties": list(reference.psd_config.confirmed_properties),
            "assumptions": list(reference.psd_config.assumptions),
        }
    return result
