"""GUI-unabhaengiger virtueller Echtzeitpfad fuer den schnellen WDF-Solver.

Eine WAV-Datei wird in festen Audiobloecken gelesen und ausschliesslich ueber
``FastBernardiniSpeakerSolver.process_block_into`` verarbeitet. Optional wird
die Blockfreigabe an die reale Abtastzeit gekoppelt. Kernelzeit, gesamte
Blockzeit, Startjitter und Deadline-Verletzungen werden getrennt erfasst.

Der wissenschaftliche Ausgang bleibt Schalldruck in Pascal und wird als
Float64-NPY gespeichert. Eine optionale PCM16-WAV ist nur ein separat
normalisierter Hoermonitor und darf nicht fuer Pegelauswertungen verwendet
werden.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable
import wave

import numpy as np
from scipy.io import wavfile

import WDFSolverFast as wdf_fast
import WDFSolverVPres as wdf_reference


@dataclass(frozen=True)
class VirtualRealtimeConfig:
    """Konfiguration eines virtuellen blockweisen Echtzeitlaufs.

    ``target_rms_voltage`` bezeichnet die tatsaechlich an den Solver uebergebene
    RMS-Spannung. Bei ``None`` bleibt die WAV-Dynamik erhalten und wird nur mit
    ``volts_per_full_scale`` von normierter Vollaussteuerung auf Volt abgebildet.
    """

    block_size: int = 128
    target_rms_voltage: float | None = 1.0
    volts_per_full_scale: float = 1.0
    paced: bool = True
    spin_wait_s: float = 0.002
    maximum_duration_s: float | None = None
    disable_gc_during_stream: bool = False
    write_monitor_wav: bool = True
    monitor_peak: float = 0.95
    output_storage_mode: str = "memmap"

    def validate(self) -> None:
        if int(self.block_size) != self.block_size or self.block_size <= 0:
            raise ValueError("Die Blockgroesse muss eine positive Ganzzahl sein.")
        if self.target_rms_voltage is not None:
            if not np.isfinite(self.target_rms_voltage) or self.target_rms_voltage <= 0.0:
                raise ValueError("Die Ziel-RMS-Spannung muss positiv und endlich sein.")
        if not np.isfinite(self.volts_per_full_scale) or self.volts_per_full_scale <= 0.0:
            raise ValueError("Volt pro Vollaussteuerung muss positiv und endlich sein.")
        if not np.isfinite(self.spin_wait_s) or self.spin_wait_s < 0.0:
            raise ValueError("Die Spin-Wartezeit darf nicht negativ sein.")
        if self.maximum_duration_s is not None:
            if not np.isfinite(self.maximum_duration_s) or self.maximum_duration_s <= 0.0:
                raise ValueError("Die maximale Dauer muss positiv und endlich sein.")
        if not np.isfinite(self.monitor_peak) or not 0.0 < self.monitor_peak <= 1.0:
            raise ValueError("Der Monitor-Spitzenwert muss innerhalb (0, 1] liegen.")
        if self.output_storage_mode not in {"memmap", "memory"}:
            raise ValueError(
                "Der Ausgabespeicher muss 'memmap' oder 'memory' sein."
            )


class WavVoltageSource:
    """Speichergemapte WAV-Quelle mit expliziter Abbildung auf Spannung."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"WAV-Eingang nicht gefunden: {self.path}")
        if self.path.suffix.lower() != ".wav":
            raise ValueError("Der virtuelle Echtzeitpfad erwartet eine WAV-Datei.")
        try:
            fs_hz, samples = wavfile.read(self.path, mmap=True)
        except ValueError:
            # Beispielsweise 24-Bit-PCM kann wegen der 3-Byte-Samplebreite
            # nicht speichergemappt werden. Die gleiche Block-API bleibt nach
            # einem konventionellen Lesevorgang dennoch erhalten.
            fs_hz, samples = wavfile.read(self.path, mmap=False)
        self.fs_hz = float(fs_hz)
        self.samples = np.asarray(samples)
        if self.samples.ndim not in (1, 2):
            raise ValueError("Die WAV-Datei muss mono oder mehrkanalig zweidimensional sein.")
        if len(self.samples) == 0:
            raise ValueError("Die WAV-Datei ist leer.")
        if not (
            np.issubdtype(self.samples.dtype, np.integer)
            or np.issubdtype(self.samples.dtype, np.floating)
        ):
            raise ValueError(f"Nicht unterstuetzter WAV-Datentyp: {self.samples.dtype}")

    @property
    def sample_count(self) -> int:
        return int(len(self.samples))

    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    def read_normalized_into(self, start: int, stop: int, destination: np.ndarray) -> None:
        """Liest einen Monoblock in normierter Vollaussteuerung als Float64."""
        source = self.samples[start:stop]
        destination = np.asarray(destination)
        expected_length = stop - start
        if destination.dtype != np.float64 or destination.shape != (expected_length,):
            raise ValueError("Der WAV-Zielpuffer muss ein passender Float64-Vektor sein.")

        if source.ndim == 1:
            np.copyto(destination, source, casting="unsafe")
        else:
            np.mean(source, axis=1, dtype=np.float64, out=destination)

        if np.issubdtype(self.samples.dtype, np.signedinteger):
            info = np.iinfo(self.samples.dtype)
            destination /= float(max(abs(info.min), abs(info.max)))
        elif np.issubdtype(self.samples.dtype, np.unsignedinteger):
            info = np.iinfo(self.samples.dtype)
            midpoint = 0.5 * (float(info.max) + float(info.min) + 1.0)
            destination -= midpoint
            destination /= midpoint

        if not np.all(np.isfinite(destination)):
            raise ValueError("Die WAV-Datei enthaelt nichtendliche Werte.")

    def normalized_rms(self, sample_count: int, block_size: int) -> float:
        scratch = np.empty(block_size, dtype=np.float64)
        energy = 0.0
        counted = 0
        for start in range(0, sample_count, block_size):
            stop = min(start + block_size, sample_count)
            block = scratch[: stop - start]
            self.read_normalized_into(start, stop, block)
            energy += float(np.dot(block, block))
            counted += len(block)
        return float(np.sqrt(energy / counted)) if counted else 0.0


class RealtimeWDFEngine:
    """Vorallokierter Blockadapter um den schnellen Bernardini-Solver."""

    def __init__(self, fs_hz: float, parameters: dict[str, Any], block_size: int):
        if block_size <= 0:
            raise ValueError("Die Blockgroesse muss positiv sein.")
        self.fs_hz = float(fs_hz)
        self.block_size = int(block_size)
        self.solver = wdf_fast.FastBernardiniSpeakerSolver(self.fs_hz, parameters)
        self.input_buffer = np.empty(self.block_size, dtype=np.float64)
        self.output_buffer = np.empty(self.block_size, dtype=np.float64)

    def reset(self) -> None:
        self.solver.reset()

    def process_block_into(
        self,
        input_voltage: np.ndarray,
        output_pressure: np.ndarray,
    ) -> np.ndarray:
        """Verarbeitet bis zu ``block_size`` Samples ohne interne Neuallokation."""
        source = np.asarray(input_voltage)
        target = np.asarray(output_pressure)
        if source.ndim != 1 or not 0 < len(source) <= self.block_size:
            raise ValueError("Der Eingangsblock muss eindimensional und hoechstens block_size lang sein.")
        if target.dtype != np.float64 or target.shape != source.shape or not target.flags.writeable:
            raise ValueError("Der Ausgangsblock muss ein beschreibbarer, gleich langer Float64-Vektor sein.")
        local_input = self.input_buffer[: len(source)]
        np.copyto(local_input, source, casting="unsafe")
        return self.solver.process_block_into(local_input, target)


class RealtimeRunCancelled(RuntimeError):
    """Signalisiert einen kontrolliert zwischen zwei Bloecken beendeten Lauf."""


class _ThreadWorkCounter:
    """Hochaufloesender Zaehler fuer die Arbeit des aktuellen Threads.

    Windows stellt mit ``QueryThreadCycleTime`` einen fuer kurze Audiobloecke
    wesentlich feineren Zaehler als ``time.thread_time()`` bereit. Auf anderen
    Plattformen dient die Thread-CPU-Zeit in Nanosekunden als Arbeitsmass.
    Die Werte werden bewusst nicht in Sekunden umgerechnet, weil CPU-Takt und
    Turbomodus waehrend eines Laufs variieren koennen.
    """

    def __init__(self) -> None:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.QueryThreadCycleTime.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulonglong),
            )
            kernel32.QueryThreadCycleTime.restype = ctypes.c_int
            self._query = kernel32.QueryThreadCycleTime
            self._thread = kernel32.GetCurrentThread()
            self.unit = "cycles"
        else:
            self._query = None
            self._thread = None
            self.unit = "ns"

    def read(self) -> float:
        if self._query is None:
            return float(time.thread_time_ns())
        value = ctypes.c_ulonglong()
        if not self._query(self._thread, ctypes.byref(value)):
            error = ctypes.get_last_error()
            raise OSError(error, "QueryThreadCycleTime ist fehlgeschlagen.")
        return float(value.value)


@dataclass(frozen=True)
class RealtimeProgress:
    completed_blocks: int
    total_blocks: int
    completed_samples: int
    total_samples: int
    latest_kernel_us: float
    latest_total_processing_us: float
    kernel_deadline_misses: int
    total_processing_deadline_misses: int
    scheduled_deadline_misses: int


@dataclass(frozen=True)
class RealtimeStatistics:
    fs_hz: float
    block_size: int
    sample_count: int
    block_count: int
    signal_duration_s: float
    wall_duration_s: float
    input_preparation_times_s: np.ndarray
    kernel_times_s: np.ndarray
    total_block_times_s: np.ndarray
    release_lateness_s: np.ndarray
    block_deadlines_s: np.ndarray
    kernel_deadline_misses: np.ndarray
    total_processing_deadline_misses: np.ndarray
    scheduled_deadline_misses: np.ndarray
    thread_work_unit: str
    input_preparation_thread_work: np.ndarray
    kernel_thread_work: np.ndarray
    total_block_thread_work: np.ndarray

    @staticmethod
    def _timing_summary(values_s: np.ndarray) -> dict[str, float]:
        values = np.asarray(values_s, dtype=float)
        if not values.size:
            return {"median_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
        return {
            "median_us": float(1e6 * np.median(values)),
            "p95_us": float(1e6 * np.percentile(values, 95)),
            "p99_us": float(1e6 * np.percentile(values, 99)),
            "max_us": float(1e6 * np.max(values)),
        }

    @staticmethod
    def _work_summary(values: np.ndarray, unit: str) -> dict[str, float | str]:
        values = np.asarray(values, dtype=float)
        if not values.size:
            return {
                "unit": unit,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "p99_over_median": 0.0,
                "max_over_median": 0.0,
            }
        median = float(np.median(values))
        p99 = float(np.percentile(values, 99))
        maximum = float(np.max(values))
        return {
            "unit": unit,
            "median": median,
            "p95": float(np.percentile(values, 95)),
            "p99": p99,
            "max": maximum,
            "p99_over_median": p99 / median if median > 0.0 else float("inf"),
            "max_over_median": maximum / median if median > 0.0 else float("inf"),
        }

    def summary(self) -> dict[str, Any]:
        total_processing_s = float(np.sum(self.total_block_times_s))
        return {
            "fs_hz": self.fs_hz,
            "block_size": self.block_size,
            "sample_count": self.sample_count,
            "block_count": self.block_count,
            "signal_duration_s": self.signal_duration_s,
            "wall_duration_s": self.wall_duration_s,
            "kernel_realtime_factor": (
                self.signal_duration_s / float(np.sum(self.kernel_times_s))
                if np.sum(self.kernel_times_s) > 0.0
                else float("inf")
            ),
            "total_processing_realtime_factor": (
                self.signal_duration_s / total_processing_s
                if total_processing_s > 0.0
                else float("inf")
            ),
            "wall_realtime_factor": (
                self.signal_duration_s / self.wall_duration_s
                if self.wall_duration_s > 0.0
                else float("inf")
            ),
            "input_preparation": self._timing_summary(self.input_preparation_times_s),
            "kernel": self._timing_summary(self.kernel_times_s),
            "total_block": self._timing_summary(self.total_block_times_s),
            "input_preparation_thread_work": self._work_summary(
                self.input_preparation_thread_work,
                self.thread_work_unit,
            ),
            "kernel_thread_work": self._work_summary(
                self.kernel_thread_work,
                self.thread_work_unit,
            ),
            "total_block_thread_work": self._work_summary(
                self.total_block_thread_work,
                self.thread_work_unit,
            ),
            "release_lateness": self._timing_summary(self.release_lateness_s),
            "kernel_deadline_misses": int(np.count_nonzero(self.kernel_deadline_misses)),
            "total_processing_deadline_misses": int(
                np.count_nonzero(self.total_processing_deadline_misses)
            ),
            "scheduled_deadline_misses": int(np.count_nonzero(self.scheduled_deadline_misses)),
        }


@dataclass
class VirtualRealtimeResult:
    input_path: Path
    preset_name: str
    fs_hz: float
    input_rms_voltage: float
    source_normalized_rms: float
    voltage_gain: float
    output_pressure_pa: np.ndarray
    output_pressure_path: Path | None
    statistics: RealtimeStatistics
    config: VirtualRealtimeConfig
    parameters: dict[str, Any]
    output_storage_mode: str
    artifact_directory: Path | None = None

    def close(self) -> None:
        """Schreibt und schliesst eine gegebenenfalls dateibasierte Druckausgabe."""
        if isinstance(self.output_pressure_pa, np.memmap):
            self.output_pressure_pa.flush()
            memory_map = getattr(self.output_pressure_pa, "_mmap", None)
            if memory_map is not None and not memory_map.closed:
                memory_map.close()

    def __enter__(self) -> "VirtualRealtimeResult":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_value: Any) -> None:
        self.close()


def _wait_until(deadline_s: float, spin_wait_s: float) -> None:
    """Wartet grob per Sleep und fuer den konfigurierten kurzen Rest aktiv.

    Unter Windows koennen Sleep-Aufrufe im Sub-ms-Bereich weit ueber die
    angeforderte Dauer hinauslaufen. Der standardmaessig aktive letzte Bereich
    von 2 ms verhindert diesen systematischen Fehler des virtuellen Taktgebers.
    Ein spaeterer Audio-Callback wird stattdessen vom Audiotreiber freigegeben.
    """
    while True:
        remaining = deadline_s - time.perf_counter()
        if remaining <= 0.0:
            return
        if remaining > spin_wait_s:
            sleep_time = remaining - spin_wait_s
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        else:
            # Der kurze aktive Rest reduziert die fuer Sub-ms-Bloecke oft zu
            # grobe Sleep-Ueberschreitung. Er ist nur Teil des virtuellen Tests.
            pass


def _limited_sample_count(source: WavVoltageSource, config: VirtualRealtimeConfig) -> int:
    if config.maximum_duration_s is None:
        return source.sample_count
    requested = int(round(config.maximum_duration_s * source.fs_hz))
    return min(source.sample_count, max(1, requested))


def _open_output_array(
    sample_count: int,
    output_directory: Path | None,
    storage_mode: str,
) -> tuple[np.ndarray, Path | None, str]:
    if storage_mode == "memory" or output_directory is None:
        return np.empty(sample_count, dtype=np.float64), None, "memory"
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "output_pressure_pa.npy"
    if path.exists():
        raise FileExistsError(f"Der wissenschaftliche Ausgang existiert bereits: {path}")
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(sample_count,))
    return output, path, "memmap"


def run_virtual_realtime_wav(
    input_path: str | Path,
    preset_name: str,
    config: VirtualRealtimeConfig | None = None,
    output_directory: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
    cancel_event: Any | None = None,
    progress_callback: Callable[[RealtimeProgress], None] | None = None,
) -> VirtualRealtimeResult:
    """Fuehrt einen blockweisen WAV-Test mit optionaler Echtzeittaktung aus."""
    active_config = config or VirtualRealtimeConfig()
    active_config.validate()
    if parameters is None:
        if preset_name not in wdf_reference.BERNARDINI_PRESETS:
            raise ValueError(f"Unbekanntes Lautsprecherpreset: {preset_name}")
        active_parameters = dict(wdf_reference.BERNARDINI_PRESETS[preset_name])
    else:
        active_parameters = dict(parameters)

    source = WavVoltageSource(input_path)
    sample_count = _limited_sample_count(source, active_config)
    source_rms = source.normalized_rms(sample_count, active_config.block_size)
    if source_rms <= np.finfo(float).tiny:
        raise ValueError("Das ausgewaehlte WAV-Signal besitzt keinen von null verschiedenen RMS-Wert.")
    if active_config.target_rms_voltage is None:
        voltage_gain = float(active_config.volts_per_full_scale)
    else:
        voltage_gain = float(active_config.target_rms_voltage / source_rms)
    input_rms_voltage = float(source_rms * voltage_gain)

    artifact_directory = (
        Path(output_directory).expanduser().resolve() if output_directory is not None else None
    )
    output, output_path, output_storage_mode = _open_output_array(
        sample_count,
        artifact_directory,
        active_config.output_storage_mode,
    )
    engine = RealtimeWDFEngine(source.fs_hz, active_parameters, active_config.block_size)

    block_count = int(math.ceil(sample_count / active_config.block_size))
    input_times = np.empty(block_count, dtype=np.float64)
    kernel_times = np.empty(block_count, dtype=np.float64)
    total_times = np.empty(block_count, dtype=np.float64)
    input_thread_work = np.empty(block_count, dtype=np.float64)
    kernel_thread_work = np.empty(block_count, dtype=np.float64)
    total_thread_work = np.empty(block_count, dtype=np.float64)
    lateness = np.empty(block_count, dtype=np.float64)
    deadlines = np.empty(block_count, dtype=np.float64)
    kernel_misses = np.empty(block_count, dtype=bool)
    total_misses = np.empty(block_count, dtype=bool)
    scheduled_misses = np.empty(block_count, dtype=bool)

    wall_start = time.perf_counter()
    scheduled_release = wall_start
    last_progress_report = wall_start
    cancelled = False
    completed_blocks = 0
    kernel_miss_count = 0
    total_miss_count = 0
    scheduled_miss_count = 0
    thread_work_counter = _ThreadWorkCounter()
    gc_was_enabled = gc.isenabled()
    if active_config.disable_gc_during_stream and gc_was_enabled:
        gc.disable()
    try:
        for block_index, start in enumerate(range(0, sample_count, active_config.block_size)):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            stop = min(start + active_config.block_size, sample_count)
            current_samples = stop - start
            block_period_s = current_samples / source.fs_hz
            deadlines[block_index] = block_period_s

            if active_config.paced:
                _wait_until(scheduled_release, active_config.spin_wait_s)
            actual_release = time.perf_counter()
            lateness[block_index] = (
                max(0.0, actual_release - scheduled_release) if active_config.paced else 0.0
            )

            total_started = actual_release
            total_work_started = thread_work_counter.read()
            input_block = engine.input_buffer[:current_samples]
            source.read_normalized_into(start, stop, input_block)
            input_block *= voltage_gain
            kernel_started = time.perf_counter()
            kernel_work_started = thread_work_counter.read()
            engine.solver.process_block_into(input_block, output[start:stop])
            completed_work = thread_work_counter.read()
            completed = time.perf_counter()

            input_times[block_index] = kernel_started - total_started
            kernel_times[block_index] = completed - kernel_started
            total_times[block_index] = completed - total_started
            input_thread_work[block_index] = kernel_work_started - total_work_started
            kernel_thread_work[block_index] = completed_work - kernel_work_started
            total_thread_work[block_index] = completed_work - total_work_started
            kernel_misses[block_index] = kernel_times[block_index] > block_period_s
            total_misses[block_index] = total_times[block_index] > block_period_s
            scheduled_deadline = (
                scheduled_release + block_period_s
                if active_config.paced
                else actual_release + block_period_s
            )
            scheduled_misses[block_index] = completed > scheduled_deadline
            kernel_miss_count += int(kernel_misses[block_index])
            total_miss_count += int(total_misses[block_index])
            scheduled_miss_count += int(scheduled_misses[block_index])
            scheduled_release += block_period_s
            completed_blocks = block_index + 1

            if progress_callback is not None and (
                completed - last_progress_report >= 0.1 or completed_blocks == block_count
            ):
                progress_callback(
                    RealtimeProgress(
                        completed_blocks=completed_blocks,
                        total_blocks=block_count,
                        completed_samples=stop,
                        total_samples=sample_count,
                        latest_kernel_us=1e6 * kernel_times[block_index],
                        latest_total_processing_us=1e6 * total_times[block_index],
                        kernel_deadline_misses=kernel_miss_count,
                        total_processing_deadline_misses=total_miss_count,
                        scheduled_deadline_misses=scheduled_miss_count,
                    )
                )
                last_progress_report = completed

        if active_config.paced and not cancelled:
            _wait_until(scheduled_release, active_config.spin_wait_s)
    finally:
        if active_config.disable_gc_during_stream and gc_was_enabled:
            gc.enable()
    if cancelled:
        if isinstance(output, np.memmap):
            output.flush()
            memory_map = getattr(output, "_mmap", None)
            if memory_map is not None and not memory_map.closed:
                memory_map.close()
        if output_path is not None and output_path.exists():
            output_path.unlink()
        raise RealtimeRunCancelled(
            f"Echtzeitlauf nach {completed_blocks} von {block_count} Bloecken abgebrochen."
        )
    wall_duration_s = time.perf_counter() - wall_start

    if isinstance(output, np.memmap):
        output.flush()
    statistics = RealtimeStatistics(
        fs_hz=source.fs_hz,
        block_size=active_config.block_size,
        sample_count=sample_count,
        block_count=block_count,
        signal_duration_s=sample_count / source.fs_hz,
        wall_duration_s=wall_duration_s,
        input_preparation_times_s=input_times,
        kernel_times_s=kernel_times,
        total_block_times_s=total_times,
        release_lateness_s=lateness,
        block_deadlines_s=deadlines,
        kernel_deadline_misses=kernel_misses,
        total_processing_deadline_misses=total_misses,
        scheduled_deadline_misses=scheduled_misses,
        thread_work_unit=thread_work_counter.unit,
        input_preparation_thread_work=input_thread_work,
        kernel_thread_work=kernel_thread_work,
        total_block_thread_work=total_thread_work,
    )
    result = VirtualRealtimeResult(
        input_path=source.path,
        preset_name=preset_name,
        fs_hz=source.fs_hz,
        input_rms_voltage=input_rms_voltage,
        source_normalized_rms=source_rms,
        voltage_gain=voltage_gain,
        output_pressure_pa=output,
        output_pressure_path=output_path,
        statistics=statistics,
        config=active_config,
        parameters=active_parameters,
        output_storage_mode=output_storage_mode,
        artifact_directory=artifact_directory,
    )
    if artifact_directory is not None:
        write_run_artifacts(result)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_block_timings(path: Path, statistics: RealtimeStatistics) -> None:
    start_samples = np.arange(statistics.block_count) * statistics.block_size
    sample_counts = np.minimum(
        statistics.block_size,
        statistics.sample_count - start_samples,
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "block_index",
                "start_sample",
                "sample_count",
                "deadline_us",
                "input_preparation_us",
                "kernel_us",
                "total_processing_us",
                f"input_preparation_thread_work_{statistics.thread_work_unit}",
                f"kernel_thread_work_{statistics.thread_work_unit}",
                f"total_processing_thread_work_{statistics.thread_work_unit}",
                "release_lateness_us",
                "kernel_deadline_miss",
                "total_processing_deadline_miss",
                "scheduled_deadline_miss",
            )
        )
        for index in range(statistics.block_count):
            writer.writerow(
                (
                    index,
                    int(start_samples[index]),
                    int(sample_counts[index]),
                    1e6 * statistics.block_deadlines_s[index],
                    1e6 * statistics.input_preparation_times_s[index],
                    1e6 * statistics.kernel_times_s[index],
                    1e6 * statistics.total_block_times_s[index],
                    statistics.input_preparation_thread_work[index],
                    statistics.kernel_thread_work[index],
                    statistics.total_block_thread_work[index],
                    1e6 * statistics.release_lateness_s[index],
                    int(statistics.kernel_deadline_misses[index]),
                    int(statistics.total_processing_deadline_misses[index]),
                    int(statistics.scheduled_deadline_misses[index]),
                )
            )


def write_monitor_wav(
    pressure_pa: np.ndarray,
    fs_hz: float,
    path: str | Path,
    target_peak: float = 0.95,
    chunk_size: int = 65536,
) -> float:
    """Schreibt einen separat normalisierten PCM16-Hoermonitor.

    Der Rueckgabewert ist der Faktor von Pascal auf normierte Audioamplitude.
    """
    signal = np.asarray(pressure_pa)
    if signal.ndim != 1:
        raise ValueError("Der Druckausgang muss eindimensional sein.")
    maximum = 0.0
    for start in range(0, len(signal), chunk_size):
        block = np.asarray(signal[start : start + chunk_size])
        if not np.all(np.isfinite(block)):
            raise ValueError("Der Druckausgang muss endlich sein.")
        maximum = max(maximum, float(np.max(np.abs(block), initial=0.0)))
    scale = float(target_peak / maximum) if maximum > np.finfo(float).tiny else 0.0
    destination = Path(path)
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(int(round(fs_hz)))
        for start in range(0, len(signal), chunk_size):
            block = np.asarray(signal[start : start + chunk_size], dtype=np.float64)
            pcm = np.rint(np.clip(block * scale, -1.0, 1.0) * 32767.0).astype("<i2")
            writer.writeframes(pcm.tobytes())
    return scale


def write_run_artifacts(result: VirtualRealtimeResult) -> None:
    """Schreibt Statistik, Metadaten und optionalen Hoermonitor nach dem Lauf."""
    directory = result.artifact_directory
    if directory is None:
        raise ValueError("Fuer Ergebnisartefakte wurde kein Ausgabeordner angegeben.")
    directory.mkdir(parents=True, exist_ok=True)

    if result.output_pressure_path is None:
        result.output_pressure_path = directory / "output_pressure_pa.npy"
        np.save(result.output_pressure_path, np.asarray(result.output_pressure_pa, dtype=np.float64))
    _write_block_timings(directory / "block_timing.csv", result.statistics)

    monitor_scale = None
    if result.config.write_monitor_wav:
        monitor_scale = write_monitor_wav(
            result.output_pressure_pa,
            result.fs_hz,
            directory / "monitor_normalized.wav",
            target_peak=result.config.monitor_peak,
        )

    metadata = {
        "schema_version": "1.2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": {
            "path": str(result.input_path),
            "sha256": _sha256(result.input_path),
            "source_normalized_rms": result.source_normalized_rms,
            "voltage_gain": result.voltage_gain,
            "input_rms_voltage": result.input_rms_voltage,
        },
        "model": {
            "preset_name": result.preset_name,
            "parameters": _jsonable(result.parameters),
            "fast_solver_version": getattr(wdf_fast, "__version__", None),
        },
        "config": _jsonable(asdict(result.config)),
        "statistics": _jsonable(result.statistics.summary()),
        "outputs": {
            "scientific_pressure": {
                "path": "output_pressure_pa.npy",
                "dtype": "float64",
                "unit": "Pa",
                "stream_buffer": result.output_storage_mode,
            },
            "block_timing": {
                "path": "block_timing.csv",
                "time_unit": "us",
            },
            "monitor_wav": (
                {
                    "path": "monitor_normalized.wav",
                    "format": "PCM16 mono",
                    "scientific_level_valid": False,
                    "pressure_pa_to_normalized_audio_scale": monitor_scale,
                }
                if monitor_scale is not None
                else None
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "notes": [
            "Die Kernelzeit umfasst ausschliesslich process_block_into().",
            "Die gesamte Blockzeit umfasst WAV-Konvertierung, Eingangsskalierung und Solver.",
            "Die Thread-Arbeit wird unter Windows mit QueryThreadCycleTime in CPU-Zyklen erfasst; auf anderen Plattformen mit thread_time_ns.",
            "Die Thread-Arbeit wird wegen dynamischer CPU-Frequenzen relativ zu ihrem Median und nicht als exakte Zeit interpretiert.",
            "Dateiabschluss, Metadaten und Monitor-WAV werden nach der zeitkritischen Schleife geschrieben.",
            "Die Monitor-WAV ist normalisiert und nicht fuer physikalische Pegelauswertungen geeignet.",
        ],
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
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


def create_unique_run_directory(
    output_root: str | Path,
    input_path: str | Path,
    preset_name: str,
    block_size: int,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    input_stem = Path(input_path).stem
    safe_stem = "".join(character if character.isalnum() or character in "-." else "-" for character in input_stem)
    base = f"{timestamp}_{preset_name}_{safe_stem}_{block_size}samples"
    candidate = root / base
    index = 2
    while candidate.exists():
        candidate = root / f"{base}_{index:03d}"
        index += 1
    candidate.mkdir()
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Virtueller blockweiser Echtzeittest des schnellen Bernardini-WDF-Solvers."
    )
    parser.add_argument("input_wav", help="WAV-Datei als Spannungsverlauf")
    parser.add_argument("--preset", choices=tuple(wdf_reference.BERNARDINI_PRESETS), default="Spk-1")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--target-rms-voltage",
        type=float,
        default=1.0,
        help="Tatsaechliche RMS-Spannung am Solver; Standard: 1 V",
    )
    parser.add_argument(
        "--preserve-wav-level",
        action="store_true",
        help="Keine RMS-Normierung; Vollaussteuerung entspricht --volts-per-full-scale",
    )
    parser.add_argument("--volts-per-full-scale", type=float, default=1.0)
    parser.add_argument("--maximum-duration", type=float, default=None)
    parser.add_argument("--unpaced", action="store_true", help="Ohne reale Zeitfreigabe verarbeiten")
    parser.add_argument("--disable-gc", action="store_true")
    parser.add_argument(
        "--output-storage",
        choices=("memory", "memmap"),
        default="memmap",
        help="RAM-Puffer mit Speicherung nach dem Lauf oder dateibasierte Memory-Map",
    )
    parser.add_argument("--no-monitor-wav", action="store_true")
    parser.add_argument("--output-root", default="data/RealtimeTests")
    args = parser.parse_args()

    output_directory = create_unique_run_directory(
        args.output_root, args.input_wav, args.preset, args.block_size
    )
    config = VirtualRealtimeConfig(
        block_size=args.block_size,
        target_rms_voltage=(None if args.preserve_wav_level else args.target_rms_voltage),
        volts_per_full_scale=args.volts_per_full_scale,
        paced=not args.unpaced,
        maximum_duration_s=args.maximum_duration,
        disable_gc_during_stream=args.disable_gc,
        write_monitor_wav=not args.no_monitor_wav,
        output_storage_mode=args.output_storage,
    )
    try:
        result = run_virtual_realtime_wav(
            args.input_wav,
            args.preset,
            config=config,
            output_directory=output_directory,
        )
    except Exception:
        # Der eindeutig bezeichnete Ordner bleibt als sichtbarer Hinweis auf den
        # fehlgeschlagenen Lauf erhalten, enthaelt aber keine Erfolgsmeldung.
        raise

    try:
        print(json.dumps(result.statistics.summary(), indent=2, ensure_ascii=False))
        print(f"Ergebnisse: {output_directory}")
    finally:
        result.close()


if __name__ == "__main__":
    main()
