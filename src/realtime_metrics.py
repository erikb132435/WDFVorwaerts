"""Gemeinsame, GUI-unabhaengige Metriken fuer Echtzeitpfade."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import sys
import time
from typing import Any

import numpy as np


def timing_summary(values_s: np.ndarray) -> dict[str, float]:
    """Fasst Zeitwerte in Sekunden als robuste Mikrosekundenkennwerte zusammen."""
    values = np.asarray(values_s, dtype=float)
    if not values.size:
        return {
            "median_us": 0.0,
            "p95_us": 0.0,
            "p99_us": 0.0,
            "p999_us": 0.0,
            "max_us": 0.0,
        }
    return {
        "median_us": float(1e6 * np.median(values)),
        "p95_us": float(1e6 * np.percentile(values, 95)),
        "p99_us": float(1e6 * np.percentile(values, 99)),
        "p999_us": float(1e6 * np.percentile(values, 99.9)),
        "max_us": float(1e6 * np.max(values)),
    }


def work_summary(values: np.ndarray, unit: str) -> dict[str, float | str]:
    """Fasst einen Threadarbeitszaehler relativ zu seinem Median zusammen."""
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


@dataclass(frozen=True)
class DeadlineEpisodes:
    """Zusammenhaengende Folgen verfehlter Blockfristen."""

    start_indices: np.ndarray
    lengths: np.ndarray
    durations_s: np.ndarray
    missed_block_count: int

    @property
    def episode_count(self) -> int:
        return int(len(self.start_indices))

    @property
    def longest_episode_blocks(self) -> int:
        return int(np.max(self.lengths, initial=0))

    @property
    def longest_episode_duration_s(self) -> float:
        return float(np.max(self.durations_s, initial=0.0))

    def summary(self) -> dict[str, int | float]:
        return {
            "missed_block_count": self.missed_block_count,
            "episode_count": self.episode_count,
            "longest_episode_blocks": self.longest_episode_blocks,
            "longest_episode_duration_ms": 1e3 * self.longest_episode_duration_s,
        }


def deadline_episodes(
    misses: np.ndarray,
    block_deadlines_s: np.ndarray | float | None = None,
) -> DeadlineEpisodes:
    """Gruppiert eine boolesche Miss-Folge in unabhaengige Episoden."""
    mask = np.asarray(misses, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("Deadline-Misses muessen eindimensional sein.")
    if not mask.size:
        empty_int = np.empty(0, dtype=np.int64)
        return DeadlineEpisodes(empty_int, empty_int.copy(), np.empty(0), 0)

    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]]).astype(np.int64)
    stops = (np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1).astype(np.int64)
    lengths = stops - starts

    if block_deadlines_s is None:
        durations = np.zeros(len(starts), dtype=np.float64)
    else:
        deadlines = np.asarray(block_deadlines_s, dtype=np.float64)
        if deadlines.ndim == 0:
            durations = lengths.astype(np.float64) * float(deadlines)
        else:
            if deadlines.shape != mask.shape:
                raise ValueError("Blockfristen und Deadline-Misses muessen gleich lang sein.")
            cumulative = np.r_[0.0, np.cumsum(deadlines)]
            durations = cumulative[stops] - cumulative[starts]

    return DeadlineEpisodes(
        start_indices=starts,
        lengths=lengths,
        durations_s=durations,
        missed_block_count=int(np.count_nonzero(mask)),
    )


class ThreadWorkCounter:
    """Hochaufloesender Arbeitszaehler fuer den aktuell ausfuehrenden Thread."""

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


def jsonable(value: Any) -> Any:
    """Konvertiert die in den Metriken verwendeten NumPy-Typen rekursiv."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Nicht JSON-serialisierbarer Wert: {type(value).__name__}")
