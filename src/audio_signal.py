"""Gemeinsame blockweise WAV-Quelle fuer virtuelle und reale Audiopfade."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


class WavVoltageSource:
    """Speichergemapte WAV-Quelle mit normierter blockweiser Monoausgabe."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"WAV-Eingang nicht gefunden: {self.path}")
        if self.path.suffix.lower() != ".wav":
            raise ValueError("Der Echtzeitpfad erwartet eine WAV-Datei.")
        try:
            fs_hz, samples = wavfile.read(self.path, mmap=True)
        except ValueError:
            # 24-Bit-PCM besitzt eine nicht speichergemapte 3-Byte-Samplebreite.
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
