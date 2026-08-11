"""Hardwareunabhaengiger Callback-Kern fuer spaetere PortAudio-Anbindung.

Die WAV-Datei wird vor dem Stream vollstaendig als normiertes Monosignal und
als physikalisches Spannungssignal vorbereitet. Der Callback selbst liest
keine Dateien, erzeugt keine Ergebnisarrays und greift nicht auf eine GUI zu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

import WDFSolverFast as wdf_fast
import WDFSolverVPres as wdf_reference
from audio_signal import WavVoltageSource
import realtime_metrics as metrics


@dataclass(frozen=True)
class AudioRealtimeConfig:
    fs_hz: int = 48000
    block_size: int = 256
    preset_name: str = "Spk-2"
    target_rms_voltage: float = 1.0
    maximum_duration_s: float | None = None
    output_mode: str = "silent"
    dry_output_gain: float = 0.1

    def validate(self) -> None:
        if self.fs_hz not in {44100, 48000}:
            raise ValueError("Der erste Audiocallback unterstuetzt 44,1 oder 48 kHz.")
        if int(self.block_size) != self.block_size or self.block_size <= 0:
            raise ValueError("Die Blockgroesse muss eine positive Ganzzahl sein.")
        if self.preset_name not in wdf_reference.BERNARDINI_PRESETS:
            raise ValueError(f"Unbekanntes Lautsprecherpreset: {self.preset_name}")
        if not np.isfinite(self.target_rms_voltage) or self.target_rms_voltage <= 0.0:
            raise ValueError("Die Ziel-RMS-Spannung muss positiv und endlich sein.")
        if self.maximum_duration_s is not None:
            if not np.isfinite(self.maximum_duration_s) or self.maximum_duration_s <= 0.0:
                raise ValueError("Die maximale Dauer muss positiv und endlich sein.")
        if self.output_mode not in {"silent", "dry"}:
            raise ValueError("Der Audioausgang muss 'silent' oder 'dry' sein.")
        if not np.isfinite(self.dry_output_gain) or not 0.0 <= self.dry_output_gain <= 1.0:
            raise ValueError("Die trockene Ausgangsverstaerkung muss in [0, 1] liegen.")


@dataclass(frozen=True)
class AudioCallbackStatus:
    input_underflow: bool = False
    input_overflow: bool = False
    output_underflow: bool = False
    output_overflow: bool = False
    priming_output: bool = False


_EMPTY_CALLBACK_STATUS = AudioCallbackStatus()


@dataclass(frozen=True)
class AudioCallbackStatistics:
    fs_hz: float
    block_size: int
    sample_count: int
    block_count: int
    callback_times_s: np.ndarray
    solver_times_s: np.ndarray
    block_deadlines_s: np.ndarray
    callback_deadline_misses: np.ndarray
    thread_work_unit: str
    solver_thread_work: np.ndarray
    input_underflows: np.ndarray
    input_overflows: np.ndarray
    output_underflows: np.ndarray
    output_overflows: np.ndarray
    priming_output_blocks: np.ndarray

    def summary(self) -> dict[str, Any]:
        episodes = metrics.deadline_episodes(
            self.callback_deadline_misses,
            self.block_deadlines_s,
        )
        signal_duration_s = self.sample_count / self.fs_hz
        callback_sum = float(np.sum(self.callback_times_s))
        solver_sum = float(np.sum(self.solver_times_s))
        return {
            "fs_hz": self.fs_hz,
            "block_size": self.block_size,
            "sample_count": self.sample_count,
            "block_count": self.block_count,
            "signal_duration_s": signal_duration_s,
            "callback": metrics.timing_summary(self.callback_times_s),
            "solver": metrics.timing_summary(self.solver_times_s),
            "solver_thread_work": metrics.work_summary(
                self.solver_thread_work,
                self.thread_work_unit,
            ),
            "callback_realtime_factor": (
                signal_duration_s / callback_sum if callback_sum > 0.0 else float("inf")
            ),
            "solver_realtime_factor": (
                signal_duration_s / solver_sum if solver_sum > 0.0 else float("inf")
            ),
            "callback_deadline_misses": int(np.count_nonzero(self.callback_deadline_misses)),
            "callback_deadline_episodes": episodes.summary(),
            "driver_status": {
                "input_underflows": int(np.count_nonzero(self.input_underflows)),
                "input_overflows": int(np.count_nonzero(self.input_overflows)),
                "output_underflows": int(np.count_nonzero(self.output_underflows)),
                "output_overflows": int(np.count_nonzero(self.output_overflows)),
                "priming_output_blocks": int(np.count_nonzero(self.priming_output_blocks)),
            },
        }


@dataclass(frozen=True)
class AudioCallbackResult:
    input_path: Path
    config: AudioRealtimeConfig
    input_rms_voltage: float
    source_normalized_rms: float
    voltage_gain: float
    input_voltage: np.ndarray
    output_pressure_pa: np.ndarray
    rendered_audio: np.ndarray
    statistics: AudioCallbackStatistics


class RealtimeAudioEngine:
    """Zustandsbehafteter, vorallokierter Callback-Prozessor ohne Audiotreiber."""

    def __init__(
        self,
        config: AudioRealtimeConfig,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.parameters = dict(
            parameters
            if parameters is not None
            else wdf_reference.BERNARDINI_PRESETS[config.preset_name]
        )
        self.solver = wdf_fast.FastBernardiniSpeakerSolver(config.fs_hz, self.parameters)
        self._thread_work_counter = metrics.ThreadWorkCounter()
        self._prepared = False
        self._completed_blocks = 0
        self._completed_samples = 0

    @property
    def is_prepared(self) -> bool:
        return self._prepared

    @property
    def is_complete(self) -> bool:
        return self._prepared and self._completed_samples >= self.sample_count

    @property
    def progress(self) -> float:
        if not self._prepared or self.sample_count == 0:
            return 0.0
        return min(1.0, self._completed_samples / self.sample_count)

    def prepare_wav(self, input_path: str | Path) -> None:
        """Laedt und skaliert das komplette Eingangssignal vor dem ersten Callback."""
        source = WavVoltageSource(input_path)
        if int(round(source.fs_hz)) != self.config.fs_hz:
            raise ValueError(
                f"WAV-Samplerate {source.fs_hz:g} Hz stimmt nicht mit "
                f"{self.config.fs_hz} Hz ueberein. Resampling erfolgt nicht im Callback."
            )
        if self.config.maximum_duration_s is None:
            sample_count = source.sample_count
        else:
            requested = int(round(self.config.maximum_duration_s * self.config.fs_hz))
            sample_count = min(source.sample_count, max(1, requested))

        normalized = np.empty(sample_count, dtype=np.float64)
        chunk_size = max(65536, self.config.block_size)
        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            source.read_normalized_into(start, stop, normalized[start:stop])
        normalized_rms = float(np.sqrt(np.mean(normalized * normalized)))
        if normalized_rms <= np.finfo(float).tiny:
            raise ValueError("Das WAV-Signal besitzt keinen von null verschiedenen RMS-Wert.")

        self.input_path = source.path
        self.sample_count = sample_count
        self.block_count = int(np.ceil(sample_count / self.config.block_size))
        self.source_normalized_rms = normalized_rms
        self.voltage_gain = float(self.config.target_rms_voltage / normalized_rms)
        self.input_rms_voltage = float(normalized_rms * self.voltage_gain)
        self.normalized_input = normalized
        self.input_voltage = normalized * self.voltage_gain
        self.dry_output = np.asarray(
            normalized * self.config.dry_output_gain,
            dtype=np.float32,
        )
        self.output_pressure_pa = np.empty(sample_count, dtype=np.float64)
        self.rendered_audio = np.empty(sample_count, dtype=np.float32)

        count = self.block_count
        self.callback_times_s = np.empty(count, dtype=np.float64)
        self.solver_times_s = np.empty(count, dtype=np.float64)
        self.block_deadlines_s = np.empty(count, dtype=np.float64)
        self.callback_deadline_misses = np.empty(count, dtype=bool)
        self.solver_thread_work = np.empty(count, dtype=np.float64)
        self.input_underflows = np.empty(count, dtype=bool)
        self.input_overflows = np.empty(count, dtype=bool)
        self.output_underflows = np.empty(count, dtype=bool)
        self.output_overflows = np.empty(count, dtype=bool)
        self.priming_output_blocks = np.empty(count, dtype=bool)
        self._completed_blocks = 0
        self._completed_samples = 0
        self.solver.reset()
        self._prepared = True

    def process_callback(
        self,
        output_buffer: np.ndarray,
        status: AudioCallbackStatus | None = None,
    ) -> bool:
        """Verarbeitet genau einen vom spaeteren Audiotreiber gelieferten Ausgangsblock.

        Der Rueckgabewert ist wahr, sobald alle gueltigen Eingangssamples
        verarbeitet wurden. Ein letzter Teilblock wird im Audioausgang mit
        Nullen aufgefuellt, ohne zusaetzliche Modellzustandssamples zu erzeugen.
        """
        if not self._prepared:
            raise RuntimeError("Vor dem Callback muss prepare_wav() aufgerufen werden.")
        if self.is_complete:
            raise RuntimeError("Der vorbereitete Audiolauf ist bereits abgeschlossen.")
        target = np.asarray(output_buffer)
        if target.ndim not in {1, 2} or len(target) != self.config.block_size:
            raise ValueError("Der Audiopuffer muss exakt block_size Frames enthalten.")
        if not np.issubdtype(target.dtype, np.floating) or not target.flags.writeable:
            raise ValueError("Der Audiopuffer muss beschreibbar und gleitkommazahlig sein.")

        callback_started = time.perf_counter()
        target.fill(0.0)
        start = self._completed_samples
        stop = min(start + self.config.block_size, self.sample_count)
        active_samples = stop - start
        if self.config.output_mode == "dry":
            if target.ndim == 1:
                target[:active_samples] = self.dry_output[start:stop]
            else:
                target[:active_samples, :] = self.dry_output[start:stop, None]

        solver_started = time.perf_counter()
        work_started = self._thread_work_counter.read()
        self.solver.process_block_into(
            self.input_voltage[start:stop],
            self.output_pressure_pa[start:stop],
        )
        work_completed = self._thread_work_counter.read()
        completed = time.perf_counter()

        self.rendered_audio[start:stop] = (
            self.dry_output[start:stop] if self.config.output_mode == "dry" else 0.0
        )
        index = self._completed_blocks
        deadline_s = self.config.block_size / self.config.fs_hz
        callback_time_s = completed - callback_started
        active_status = status or _EMPTY_CALLBACK_STATUS
        self.callback_times_s[index] = callback_time_s
        self.solver_times_s[index] = completed - solver_started
        self.block_deadlines_s[index] = deadline_s
        self.callback_deadline_misses[index] = callback_time_s > deadline_s
        self.solver_thread_work[index] = work_completed - work_started
        self.input_underflows[index] = active_status.input_underflow
        self.input_overflows[index] = active_status.input_overflow
        self.output_underflows[index] = active_status.output_underflow
        self.output_overflows[index] = active_status.output_overflow
        self.priming_output_blocks[index] = active_status.priming_output
        self._completed_blocks += 1
        self._completed_samples = stop
        return self.is_complete

    def result(self) -> AudioCallbackResult:
        if not self.is_complete:
            raise RuntimeError("Ein Ergebnis ist erst nach dem letzten Callback verfuegbar.")
        count = self._completed_blocks
        statistics = AudioCallbackStatistics(
            fs_hz=float(self.config.fs_hz),
            block_size=self.config.block_size,
            sample_count=self.sample_count,
            block_count=count,
            callback_times_s=self.callback_times_s[:count].copy(),
            solver_times_s=self.solver_times_s[:count].copy(),
            block_deadlines_s=self.block_deadlines_s[:count].copy(),
            callback_deadline_misses=self.callback_deadline_misses[:count].copy(),
            thread_work_unit=self._thread_work_counter.unit,
            solver_thread_work=self.solver_thread_work[:count].copy(),
            input_underflows=self.input_underflows[:count].copy(),
            input_overflows=self.input_overflows[:count].copy(),
            output_underflows=self.output_underflows[:count].copy(),
            output_overflows=self.output_overflows[:count].copy(),
            priming_output_blocks=self.priming_output_blocks[:count].copy(),
        )
        return AudioCallbackResult(
            input_path=self.input_path,
            config=self.config,
            input_rms_voltage=self.input_rms_voltage,
            source_normalized_rms=self.source_normalized_rms,
            voltage_gain=self.voltage_gain,
            input_voltage=self.input_voltage.copy(),
            output_pressure_pa=self.output_pressure_pa.copy(),
            rendered_audio=self.rendered_audio.copy(),
            statistics=statistics,
        )


class SimulatedAudioBackend:
    """Ruft den Callback wie ein blockweiser Ausgangstreiber ohne Wartezeit auf."""

    def __init__(
        self,
        status_by_block: Mapping[int, AudioCallbackStatus] | None = None,
    ) -> None:
        self.status_by_block = dict(status_by_block or {})

    def run(self, engine: RealtimeAudioEngine) -> AudioCallbackResult:
        if not engine.is_prepared:
            raise RuntimeError("Die Audioengine muss vor dem Backendlauf vorbereitet sein.")
        output = np.empty((engine.config.block_size, 1), dtype=np.float32)
        block_index = 0
        while not engine.is_complete:
            engine.process_callback(
                output,
                status=self.status_by_block.get(block_index),
            )
            block_index += 1
        return engine.result()


def run_simulated_wav(
    input_path: str | Path,
    config: AudioRealtimeConfig,
    parameters: dict[str, Any] | None = None,
    status_by_block: Mapping[int, AudioCallbackStatus] | None = None,
) -> AudioCallbackResult:
    """Komfortfunktion fuer einen vollstaendigen hardwarefreien Callbacklauf."""
    engine = RealtimeAudioEngine(config, parameters=parameters)
    engine.prepare_wav(input_path)
    return SimulatedAudioBackend(status_by_block=status_by_block).run(engine)
