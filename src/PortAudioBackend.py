"""Realer PortAudio-Ausgangsadapter fuer den hardwareunabhaengigen Callbackkern.

Der Adapter importiert ``sounddevice`` optional. Dadurch bleiben Solver,
Simulation, GUI und hardwarefreie Tests auch ohne installierte Audioabhaengigkeit
nutzbar. Im Audiocallback selbst werden keine Dateien geschrieben.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

import RealtimeAudioEngine as audio_engine
import WDFSolverVPres as wdf_reference
from audio_signal import WavVoltageSource
import realtime_metrics as metrics


class PortAudioDependencyError(RuntimeError):
    """Das optionale Python-Paket ``sounddevice`` ist nicht verfuegbar."""


class PortAudioRunCancelled(RuntimeError):
    """Ein Hardwarelauf wurde kontrolliert ausserhalb des Callbacks beendet."""


@dataclass(frozen=True)
class PortAudioBackendStatistics:
    adapter_callback_times_s: np.ndarray
    block_deadlines_s: np.ndarray
    adapter_deadline_misses: np.ndarray
    actual_samplerate_hz: float
    requested_block_size: int
    stream_block_size: int
    stream_latency_s: float
    wasapi_exclusive: bool
    device: dict[str, Any]
    hostapi: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        episodes = metrics.deadline_episodes(
            self.adapter_deadline_misses,
            self.block_deadlines_s,
        )
        return {
            "actual_samplerate_hz": self.actual_samplerate_hz,
            "requested_block_size": self.requested_block_size,
            "stream_block_size": self.stream_block_size,
            "stream_latency_ms": 1e3 * self.stream_latency_s,
            "wasapi_exclusive": self.wasapi_exclusive,
            "adapter_callback": metrics.timing_summary(self.adapter_callback_times_s),
            "adapter_deadline_misses": int(np.count_nonzero(self.adapter_deadline_misses)),
            "adapter_deadline_episodes": episodes.summary(),
            "device": self.device,
            "hostapi": self.hostapi,
        }


@dataclass(frozen=True)
class PortAudioRunResult:
    audio: audio_engine.AudioCallbackResult
    backend: PortAudioBackendStatistics


def _load_sounddevice(module: Any | None = None) -> Any:
    if module is not None:
        return module
    try:
        import sounddevice as sounddevice_module
    except ImportError as exc:
        raise PortAudioDependencyError(
            "Das optionale Paket 'sounddevice' fehlt. Installiere es in derselben "
            "Python-Umgebung, mit der der Hardwaretest gestartet wird."
        ) from exc
    return sounddevice_module


def _plain_mapping(value: Any) -> dict[str, Any]:
    return {str(key): metrics.jsonable(item) for key, item in dict(value).items()}


def list_output_devices(sounddevice_module: Any | None = None) -> list[dict[str, Any]]:
    """Liefert alle PortAudio-Geraete mit mindestens einem Ausgangskanal."""
    sd = _load_sounddevice(sounddevice_module)
    hostapis = sd.query_hostapis()
    devices = []
    for index, raw_device in enumerate(sd.query_devices()):
        device = _plain_mapping(raw_device)
        if int(device.get("max_output_channels", 0)) < 1:
            continue
        hostapi_index = int(device.get("hostapi", -1))
        hostapi = (
            _plain_mapping(hostapis[hostapi_index])
            if 0 <= hostapi_index < len(hostapis)
            else {}
        )
        devices.append(
            {
                "index": index,
                "name": str(device.get("name", "")),
                "hostapi": str(hostapi.get("name", "")),
                "max_output_channels": int(device.get("max_output_channels", 0)),
                "default_samplerate": float(device.get("default_samplerate", 0.0)),
                "default_low_output_latency_s": float(
                    device.get("default_low_output_latency", 0.0)
                ),
                "default_high_output_latency_s": float(
                    device.get("default_high_output_latency", 0.0)
                ),
            }
        )
    return devices


def _device_details(sd: Any, device: int | str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_device = sd.query_devices(device, kind="output")
    device_info = _plain_mapping(raw_device)
    hostapi_index = int(device_info.get("hostapi", -1))
    hostapi_info = (
        _plain_mapping(sd.query_hostapis(hostapi_index)) if hostapi_index >= 0 else {}
    )
    return device_info, hostapi_info


def _map_status(status: Any) -> audio_engine.AudioCallbackStatus:
    return audio_engine.AudioCallbackStatus(
        input_underflow=bool(getattr(status, "input_underflow", False)),
        input_overflow=bool(getattr(status, "input_overflow", False)),
        output_underflow=bool(getattr(status, "output_underflow", False)),
        output_overflow=bool(getattr(status, "output_overflow", False)),
        priming_output=bool(getattr(status, "priming_output", False)),
    )


class SoundDevicePortAudioBackend:
    """Verbindet ``RealtimeAudioEngine`` mit einem sounddevice-OutputStream."""

    def __init__(self, sounddevice_module: Any | None = None) -> None:
        self.sd = _load_sounddevice(sounddevice_module)

    def run(
        self,
        engine: audio_engine.RealtimeAudioEngine,
        device: int | str | None = None,
        latency: str | float = "low",
        wasapi_exclusive: bool = False,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[float], None] | None = None,
        progress_interval_s: float = 0.25,
    ) -> PortAudioRunResult:
        if not engine.is_prepared:
            raise RuntimeError("Die Audioengine muss vor dem PortAudio-Stream vorbereitet sein.")
        if progress_interval_s <= 0.0:
            raise ValueError("Das Fortschrittsintervall muss positiv sein.")

        config = engine.config
        device_info, hostapi_info = _device_details(self.sd, device)
        hostapi_name = str(hostapi_info.get("name", ""))
        if wasapi_exclusive and "WASAPI" not in hostapi_name.upper():
            raise ValueError(
                "WASAPI Exclusive kann nur mit einem Windows-WASAPI-Geraet "
                "verwendet werden."
            )
        self.sd.check_output_settings(
            device=device,
            channels=1,
            dtype="float32",
            samplerate=config.fs_hz,
        )
        extra_settings = (
            self.sd.WasapiSettings(exclusive=True) if wasapi_exclusive else None
        )
        adapter_times = np.empty(engine.block_count, dtype=np.float64)
        deadlines = np.full(
            engine.block_count,
            config.block_size / config.fs_hz,
            dtype=np.float64,
        )
        adapter_misses = np.empty(engine.block_count, dtype=bool)
        completed_callbacks = 0
        stop_requested = False
        callback_error: list[BaseException] = []
        finished_event = threading.Event()

        def finished_callback() -> None:
            finished_event.set()

        def callback(outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            nonlocal completed_callbacks, stop_requested
            started = time.perf_counter()

            # Einige PortAudio-Backends fordern waehrend des asynchronen
            # Streamabschlusses noch einen Callback an. Dieser gehoert nicht
            # mehr zum Modelllauf und darf die bereits abgeschlossene Engine
            # deshalb nicht erneut aufrufen.
            if stop_requested or engine.is_complete:
                outdata.fill(0.0)
                stop_requested = True
                raise self.sd.CallbackStop

            index = completed_callbacks
            try:
                if frames != config.block_size:
                    outdata.fill(0.0)
                    raise RuntimeError(
                        f"PortAudio lieferte {frames} statt {config.block_size} Frames."
                    )
                complete = engine.process_callback(outdata, _map_status(status))
            except BaseException as exc:
                if index < len(adapter_times):
                    elapsed = time.perf_counter() - started
                    adapter_times[index] = elapsed
                    adapter_misses[index] = elapsed > deadlines[index]
                callback_error.append(exc)
                raise self.sd.CallbackAbort from exc

            elapsed = time.perf_counter() - started
            adapter_times[index] = elapsed
            adapter_misses[index] = elapsed > deadlines[index]
            completed_callbacks += 1
            if complete:
                stop_requested = True
                raise self.sd.CallbackStop

        timeout_s = engine.sample_count / config.fs_hz + max(10.0, 0.1 * engine.sample_count / config.fs_hz)
        stream_arguments = {
            "samplerate": config.fs_hz,
            "blocksize": config.block_size,
            "device": device,
            "channels": 1,
            "dtype": "float32",
            "latency": latency,
            "callback": callback,
            "finished_callback": finished_callback,
        }
        if extra_settings is not None:
            stream_arguments["extra_settings"] = extra_settings
        stream = self.sd.OutputStream(
            **stream_arguments,
        )
        started_wait = time.perf_counter()
        last_progress = started_wait
        try:
            with stream:
                while not finished_event.wait(0.05):
                    if cancel_event is not None and cancel_event.is_set():
                        raise PortAudioRunCancelled("Der PortAudio-Lauf wurde abgebrochen.")
                    now = time.perf_counter()
                    if progress_callback is not None and now - last_progress >= progress_interval_s:
                        progress_callback(engine.progress)
                        last_progress = now
                    if now - started_wait > timeout_s:
                        raise TimeoutError("Der PortAudio-Stream wurde nicht rechtzeitig beendet.")
        finally:
            if progress_callback is not None:
                progress_callback(engine.progress)

        if callback_error:
            raise RuntimeError("Fehler im PortAudio-Callback.") from callback_error[0]
        if not engine.is_complete:
            raise RuntimeError("PortAudio endete vor dem letzten vorbereiteten Sample.")

        audio_result = engine.result()
        count = completed_callbacks
        backend_statistics = PortAudioBackendStatistics(
            adapter_callback_times_s=adapter_times[:count].copy(),
            block_deadlines_s=deadlines[:count].copy(),
            adapter_deadline_misses=adapter_misses[:count].copy(),
            actual_samplerate_hz=float(getattr(stream, "samplerate", config.fs_hz)),
            requested_block_size=config.block_size,
            stream_block_size=int(getattr(stream, "blocksize", config.block_size)),
            stream_latency_s=float(getattr(stream, "latency", 0.0)),
            wasapi_exclusive=wasapi_exclusive,
            device=device_info,
            hostapi=hostapi_info,
        )
        return PortAudioRunResult(audio=audio_result, backend=backend_statistics)


def run_portaudio_wav(
    input_path: str | Path,
    config: audio_engine.AudioRealtimeConfig,
    device: int | str | None = None,
    latency: str | float = "low",
    wasapi_exclusive: bool = False,
    parameters: dict[str, Any] | None = None,
    sounddevice_module: Any | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> PortAudioRunResult:
    engine = audio_engine.RealtimeAudioEngine(config, parameters=parameters)
    engine.prepare_wav(input_path)
    backend = SoundDevicePortAudioBackend(sounddevice_module=sounddevice_module)
    return backend.run(
        engine,
        device=device,
        latency=latency,
        wasapi_exclusive=wasapi_exclusive,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_unique_hardware_run_directory(
    output_root: str | Path,
    result: PortAudioRunResult,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    device_name = str(result.backend.device.get("name", "default"))
    safe_device = "".join(
        character if character.isalnum() or character in "-." else "-"
        for character in device_name
    ).strip("-") or "default"
    config = result.audio.config
    base = f"{timestamp}_{config.fs_hz}Hz_{config.block_size}samples_{safe_device}"
    candidate = root / base
    index = 2
    while candidate.exists():
        candidate = root / f"{base}_{index:03d}"
        index += 1
    candidate.mkdir()
    return candidate


def write_portaudio_artifacts(
    result: PortAudioRunResult,
    output_directory: str | Path,
    sounddevice_module: Any | None = None,
) -> Path:
    """Speichert Druck und Laufzeitdaten ausschliesslich nach Streamende."""
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pressure_path = directory / "output_pressure_pa.npy"
    timing_path = directory / "callback_timing.csv"
    metadata_path = directory / "run_metadata.json"
    for path in (pressure_path, timing_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Hardwaretestergebnis existiert bereits: {path}")

    np.save(pressure_path, result.audio.output_pressure_pa)
    engine_stats = result.audio.statistics
    backend_stats = result.backend
    start_samples = np.arange(engine_stats.block_count) * engine_stats.block_size
    sample_counts = np.minimum(
        engine_stats.block_size,
        engine_stats.sample_count - start_samples,
    )
    with timing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "block_index",
                "start_sample",
                "sample_count",
                "deadline_us",
                "adapter_callback_us",
                "engine_callback_us",
                "solver_us",
                "adapter_deadline_miss",
                "engine_deadline_miss",
                "input_underflow",
                "input_overflow",
                "output_underflow",
                "output_overflow",
                "priming_output",
            )
        )
        for index in range(engine_stats.block_count):
            writer.writerow(
                (
                    index,
                    int(start_samples[index]),
                    int(sample_counts[index]),
                    1e6 * backend_stats.block_deadlines_s[index],
                    1e6 * backend_stats.adapter_callback_times_s[index],
                    1e6 * engine_stats.callback_times_s[index],
                    1e6 * engine_stats.solver_times_s[index],
                    int(backend_stats.adapter_deadline_misses[index]),
                    int(engine_stats.callback_deadline_misses[index]),
                    int(engine_stats.input_underflows[index]),
                    int(engine_stats.input_overflows[index]),
                    int(engine_stats.output_underflows[index]),
                    int(engine_stats.output_overflows[index]),
                    int(engine_stats.priming_output_blocks[index]),
                )
            )

    sd = _load_sounddevice(sounddevice_module)
    portaudio_version = None
    if hasattr(sd, "get_portaudio_version"):
        portaudio_version = metrics.jsonable(sd.get_portaudio_version())
    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": {
            "path": str(result.audio.input_path),
            "sha256": _sha256(result.audio.input_path),
            "source_normalized_rms": result.audio.source_normalized_rms,
            "voltage_gain": result.audio.voltage_gain,
            "input_rms_voltage": result.audio.input_rms_voltage,
        },
        "config": metrics.jsonable(asdict(result.audio.config)),
        "audio_output": {
            "mode": result.audio.config.output_mode,
            "dry_output_gain": result.audio.config.dry_output_gain,
            "pressure_is_audio_output": False,
            "scaled_pressure_monitor_is_audio_output": (
                result.audio.config.output_mode == "pressure"
            ),
            "monitor_diagnostics": metrics.jsonable(result.audio.output_summary()),
        },
        "engine_statistics": metrics.jsonable(engine_stats.summary()),
        "portaudio_statistics": metrics.jsonable(backend_stats.summary()),
        "outputs": {
            "scientific_pressure": {
                "path": pressure_path.name,
                "dtype": "float64",
                "unit": "Pa",
            },
            "callback_timing": {
                "path": timing_path.name,
                "time_unit": "us",
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sounddevice": getattr(sd, "__version__", None),
            "portaudio": portaudio_version,
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "notes": [
            "Alle Dateien wurden erst nach Ende des PortAudio-Streams geschrieben.",
            "Der wissenschaftliche Ausgang ist Schalldruck in Pascal und kein direktes Audiosignal.",
            "Der sichere Standardmodus gibt waehrend des Hardwaretests Stille aus.",
            "Der optionale Druckmonitor verwendet eine feste Skalierung und harte Spitzenbegrenzung.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return directory


def _parse_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _parse_latency(value: str) -> str | float:
    if value in {"low", "high"}:
        return value
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("Die Latenz muss positiv sein.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless-PortAudio-Test des schnellen WDF-Vorwaertsmodells."
    )
    parser.add_argument("input_wav", nargs="?", help="WAV-Datei mit 44,1 oder 48 kHz")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", help="PortAudio-Geraeteindex oder Namensfragment")
    parser.add_argument("--sample-rate", type=int, choices=(44100, 48000))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--preset",
        choices=tuple(wdf_reference.BERNARDINI_PRESETS),
        default="Spk-2",
    )
    parser.add_argument("--target-rms-voltage", type=float, default=3.0)
    parser.add_argument("--maximum-duration", type=float)
    parser.add_argument("--latency", type=_parse_latency, default="low")
    parser.add_argument(
        "--wasapi-exclusive",
        action="store_true",
        help="WASAPI-Geraet exklusiv statt im Windows-Shared-Modus oeffnen",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--dry-output", action="store_true")
    output_group.add_argument(
        "--pressure-output",
        action="store_true",
        help="skalierten und begrenzten Modelldruck hoerbar ausgeben",
    )
    parser.add_argument("--dry-gain", type=float, default=0.1)
    parser.add_argument(
        "--pressure-pa-per-full-scale",
        type=float,
        default=20.0,
        help="Pascal, die vor der Schutzbegrenzung digitalem Vollpegel entsprechen",
    )
    parser.add_argument(
        "--output-peak-limit",
        type=float,
        default=0.1,
        help="harte digitale Spitzengrenze in (0, 1], Standard 0.1 = -20 dBFS",
    )
    parser.add_argument("--output-root", default="data/AudioCallbackTests")
    args = parser.parse_args()

    if args.list_devices:
        for device in list_output_devices():
            print(
                f"{device['index']:>3}: [{device['hostapi']}] {device['name']} | "
                f"out={device['max_output_channels']} | "
                f"default={device['default_samplerate']:g} Hz"
            )
        if args.input_wav is None:
            return
    if args.input_wav is None:
        parser.error("input_wav fehlt; alternativ nur --list-devices verwenden.")

    source = WavVoltageSource(args.input_wav)
    sample_rate = args.sample_rate or int(round(source.fs_hz))
    output_mode = (
        "pressure" if args.pressure_output else "dry" if args.dry_output else "silent"
    )
    config = audio_engine.AudioRealtimeConfig(
        fs_hz=sample_rate,
        block_size=args.block_size,
        preset_name=args.preset,
        target_rms_voltage=args.target_rms_voltage,
        maximum_duration_s=args.maximum_duration,
        output_mode=output_mode,
        dry_output_gain=args.dry_gain,
        pressure_pa_per_full_scale=args.pressure_pa_per_full_scale,
        output_peak_limit=args.output_peak_limit,
    )
    config.validate()
    device = _parse_device(args.device)

    print(
        f"Starte {output_mode}-Hardwaretest: "
        f"{sample_rate} Hz, {args.block_size} Samples, {args.preset}."
    )
    if args.dry_output:
        print("ACHTUNG: Der trockene Audioausgang ist aktiviert. Ausgangspegel niedrig einstellen.")
    if args.pressure_output:
        print(
            "ACHTUNG: Der skalierte Modelldruck wird hoerbar ausgegeben. "
            "Interfacepegel vor dem Start auf Minimum stellen."
        )
        print(
            f"Skalierung: {args.pressure_pa_per_full_scale:g} Pa/FS, "
            f"Spitzengrenze: +/-{args.output_peak_limit:g}."
        )

    def report(progress: float) -> None:
        print(f"Fortschritt: {100.0 * progress:5.1f}%", end="\r", flush=True)

    try:
        result = run_portaudio_wav(
            args.input_wav,
            config,
            device=device,
            latency=args.latency,
            wasapi_exclusive=args.wasapi_exclusive,
            progress_callback=report,
        )
    except KeyboardInterrupt as exc:
        raise PortAudioRunCancelled("Hardwaretest durch Benutzer abgebrochen.") from exc
    print()
    directory = create_unique_hardware_run_directory(args.output_root, result)
    write_portaudio_artifacts(result, directory)
    print(json.dumps(result.backend.summary(), indent=2, ensure_ascii=False))
    print(json.dumps(result.audio.statistics.summary(), indent=2, ensure_ascii=False))
    print(json.dumps(result.audio.output_summary(), indent=2, ensure_ascii=False))
    print(f"Ergebnisse: {directory}")


if __name__ == "__main__":
    main()
