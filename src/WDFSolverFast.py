"""Skalarer, blockfaehiger Parallel-Solver fuer das Bernardini-WDF-Modell.

``WDFSolverVPres`` bleibt die wissenschaftliche Referenzimplementierung. Dieses
Modul bildet denselben Rechenpfad fuer die Bernardini-Polynommodelle ab, haelt
die zeitkritischen Groessen jedoch als Skalare und erzeugt innerhalb der
Sampleschleife keine kleinen NumPy-Arrays oder Streumatrizen.

Der Solver arbeitet weiterhin vollstaendig in Python/Float64. Die Block-API ist
so gehalten, dass der Rechenkern in einem spaeteren Schritt kompiliert werden
kann, ohne die Ein- und Ausgabe der aufrufenden Anwendung zu aendern.
"""

from __future__ import annotations

from typing import Any

import numpy as np


TRACE_KEYS = (
    "input_voltage",
    "output_pressure",
    "displacement",
    "diaphragm_velocity",
    "volume_velocity",
    "voice_coil_current",
)

__version__ = "0.2.0"


def _positive_float(value: Any, name: str, minimum: float = 0.0) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= minimum:
        raise ValueError(f"{name} muss groesser als {minimum:g} sein; erhalten: {result!r}.")
    return result


def _quartic_coefficients(params: dict[str, Any], stem: str) -> tuple[float, ...]:
    """Liest ein Polynom bis Grad vier und fuellt fehlende Ordnungen mit null."""
    key = f"{stem}_coeffs"
    if key not in params:
        raise ValueError(
            f"Der schnelle Solver benoetigt `{key}`; freie Kennlinienfunktionen "
            "werden im Referenzsolver ausgefuehrt."
        )
    coefficients = np.asarray(params[key], dtype=np.float64)
    if coefficients.ndim != 1 or not 1 <= len(coefficients) <= 5:
        raise ValueError(f"`{key}` muss ein eindimensionales Polynom bis Grad vier enthalten.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"`{key}` enthaelt nichtendliche Koeffizienten.")
    padded = np.zeros(5, dtype=np.float64)
    padded[: len(coefficients)] = coefficients
    return tuple(float(value) for value in padded)


class FastBernardiniSpeakerSolver:
    """Zustandsbehafteter skalarer Blocksolver parallel zum Referenzmodell.

    Die Parameterstruktur entspricht ``BernardiniSpeakerSolver(fs, params)``.
    Fuer den schnellen Pfad muessen ``Bl``, ``Le`` und ``Kms`` als Polynome bis
    zum Grad vier vorliegen. ``process_block`` ist der vorgesehene Rechenpfad;
    ``step`` bleibt als kompatible, aber nicht laufzeitoptimierte Einzelsample-
    Schnittstelle verfuegbar.
    """

    def __init__(self, fs: float, params: dict[str, Any]):
        self.fs = _positive_float(fs, "fs")
        self.ts = 1.0 / self.fs

        self.Re = _positive_float(params["Re"], "Re")
        self.Mms = _positive_float(params["Mms"], "Mms")
        self.Rms = _positive_float(params["Rms"], "Rms")
        self.Sd = _positive_float(params["Sd"], "Sd")
        self.Rcab = _positive_float(params.get("Rcab", 1e6), "Rcab")
        self.Ccab = _positive_float(params.get("Ccab", 1e-12), "Ccab")
        self.Ral = _positive_float(params.get("Ral", 1e6), "Ral")

        self.xi_x = float(params.get("xi_x", 0.9999))
        self.xi_le = float(params.get("xi_Le", 0.9999))
        self.xi_cms = float(params.get("xi_Cms", 0.9999))
        for name in ("xi_x", "xi_le", "xi_cms"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} muss endlich sein.")

        self.bl_coeffs = _quartic_coefficients(params, "Bl")
        self.le_coeffs = _quartic_coefficients(params, "Le")
        self.kms_coeffs = _quartic_coefficients(params, "Kms")

        # Konstante Portwiderstaende.
        self.z8 = self.Mms * self.fs
        self.z9 = self.Rms
        self.z11 = self.Rcab
        self.z12 = self.ts / self.Ccab
        self.z15 = self.Ral

        self.reset()

    def reset(self) -> None:
        """Setzt alle dynamischen Zustaende wie eine neue Referenzinstanz zurueck."""
        self.x_prev2 = 0.0
        self.x_prev1 = 0.0
        self.ims_prev = 0.0

        # Vorherige Wellen der vier dynamischen Einports.
        self.a5_prev = 0.0
        self.b5_prev = 0.0
        self.a7_prev = 0.0
        self.b7_prev = 0.0
        self.a8_prev = 0.0
        self.b8_prev = 0.0
        self.a12_prev = 0.0
        self.b12_prev = 0.0

        # Le startet im Referenzsolver mit Z=1; die anderen variablen Ports
        # werden vor ihrer ersten Reflexion vollstaendig aktualisiert.
        self.z5 = 1.0
        self.z7 = 1.0

        self.last_output_pressure = 0.0
        self.last_volume_velocity = 0.0
        self.last_displacement = 0.0
        self.last_current = 0.0
        self.bl = float(self.bl_coeffs[0])

    def step(self, vin: float) -> float:
        """Berechnet ein Sample kompatibel zur Referenz-API.

        Fuer hohe Leistung ist ``process_block`` zu verwenden, da dieser Aufruf
        absichtlich nur die Schnittstellenkompatibilitaet sicherstellt.
        """
        block = np.asarray([vin], dtype=np.float64)
        return float(self.process_block(block)[0])

    def process_block(
        self,
        input_signal: np.ndarray,
        record_traces: bool = False,
        output_buffer: np.ndarray | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Verarbeitet einen Block und behaelt den Endzustand fuer den Folgeblock.

        ``output_buffer`` erlaubt dem spaeteren Audiopfad, einen vorhandenen
        Float64-Puffer wiederzuverwenden. Ohne Puffer bleibt das Rueckgabeverhalten
        kompatibel zu ``WDFSolverVPres.simulate_model``.
        """
        input_array = np.asarray(input_signal, dtype=np.float64)
        if input_array.ndim != 1:
            raise ValueError("Das Eingangssignal muss eindimensional sein.")

        sample_count = len(input_array)
        if output_buffer is None:
            output = np.empty(sample_count, dtype=np.float64)
        else:
            output = np.asarray(output_buffer)
            if output.dtype != np.float64:
                raise ValueError("Der Ausgangspuffer muss den Datentyp Float64 besitzen.")
            if output.ndim != 1 or len(output) != sample_count:
                raise ValueError("Der Ausgangspuffer muss eindimensional und gleich lang wie der Eingang sein.")
            if not output.flags.writeable:
                raise ValueError("Der Ausgangspuffer muss beschreibbar sein.")
        if record_traces:
            displacement_trace = np.empty(sample_count, dtype=np.float64)
            diaphragm_velocity_trace = np.empty(sample_count, dtype=np.float64)
            volume_velocity_trace = np.empty(sample_count, dtype=np.float64)
            voice_coil_current_trace = np.empty(sample_count, dtype=np.float64)
        else:
            displacement_trace = None
            diaphragm_velocity_trace = None
            volume_velocity_trace = None
            voice_coil_current_trace = None

        # Unveraenderliche Parameter und Polynomkoeffizienten werden einmal pro
        # Block lokal gebunden. Dadurch entfallen die meisten Attributzugriffe
        # innerhalb der zeitkritischen Sampleschleife.
        ts = self.ts
        re = self.Re
        rms = self.Rms
        sd = self.Sd
        sd_squared = sd * sd
        xi_x = self.xi_x
        xi_le = self.xi_le
        xi_cms = self.xi_cms
        z8 = self.z8
        z9 = self.z9
        z11 = self.z11
        z12 = self.z12
        z15 = self.z15

        bl0, bl1, bl2, bl3, bl4 = self.bl_coeffs
        le0, le1, le2, le3, le4 = self.le_coeffs
        kms0, kms1, kms2, kms3, kms4 = self.kms_coeffs

        x_prev2 = self.x_prev2
        x_prev1 = self.x_prev1
        ims_prev = self.ims_prev
        a5_prev = self.a5_prev
        b5_prev = self.b5_prev
        a7_prev = self.a7_prev
        b7_prev = self.b7_prev
        a8_prev = self.a8_prev
        b8_prev = self.b8_prev
        a12_prev = self.a12_prev
        b12_prev = self.b12_prev
        z5 = self.z5

        last_current = self.last_current
        last_pressure = self.last_output_pressure
        last_volume_velocity = self.last_volume_velocity
        last_displacement = self.last_displacement
        bl = self.bl

        for index in range(sample_count):
            vin = float(input_array[index])
            x_hat = x_prev1

            # Horner-Auswertung der drei Kennlinien und ihrer Ableitungen.
            bl = (((bl4 * x_hat + bl3) * x_hat + bl2) * x_hat + bl1) * x_hat + bl0
            le = (((le4 * x_hat + le3) * x_hat + le2) * x_hat + le1) * x_hat + le0
            le_prime = ((4.0 * le4 * x_hat + 3.0 * le3) * x_hat + 2.0 * le2) * x_hat + le1
            kms = (((kms4 * x_hat + kms3) * x_hat + kms2) * x_hat + kms1) * x_hat + kms0
            kms_prime = (
                (4.0 * kms4 * x_hat + 3.0 * kms3) * x_hat + 2.0 * kms2
            ) * x_hat + kms1

            if le <= 1e-9:
                raise ValueError(f"Le(x) muss groesser als 1e-9 sein; erhalten: {le!r}.")
            if kms <= 1e-9:
                raise ValueError(f"Kms(x) muss groesser als 1e-9 sein; erhalten: {kms!r}.")

            z5_prev = z5
            z5 = le_prime * ims_prev + le / ts
            cms_prime = -kms_prime / (kms * kms)
            previous_kms_voltage = 0.5 * (a7_prev + b7_prev)
            z7 = kms * ts * (1.0 - cms_prime * previous_kms_voltage)
            if z5 <= 1e-9:
                raise ValueError(
                    f"zeitvarianter Le-Portwiderstand muss groesser als 1e-9 sein; erhalten: {z5!r}."
                )
            if z7 <= 1e-9:
                raise ValueError(
                    f"zeitvarianter Kms-Portwiderstand muss groesser als 1e-9 sein; erhalten: {z7!r}."
                )

            # Aufwaertspfad der angepassten Einports und Adaptoren.
            b4 = vin
            b5 = xi_le * (b5_prev - a5_prev) * le / (2.0 * ts * z5_prev)
            b7 = 0.5 * xi_cms * (a7_prev + b7_prev)
            b8 = 0.5 * (b8_prev - a8_prev)
            b12 = 0.5 * (b12_prev + a12_prev)

            b6 = -b4 - b5
            b10 = -b7 - b8
            b13 = -b12
            z13 = z11 + z12
            parallel_sum = z13 + z15
            b16 = (z15 / parallel_sum) * b13

            z1 = re + z5
            z2 = z7 + z8 + z9
            z3 = (z13 * z15) / parallel_sum
            bl_squared = bl * bl
            denominator = bl_squared + z1 * z3 * sd_squared + z1 * z2
            rho = 1.0 / denominator

            # Nichtreziproke 3-Port-Root, Gl. (8) bei Bernardini et al.
            a6 = rho * (
                (bl_squared - z1 * z3 * sd_squared - z1 * z2) * b6
                - 2.0 * bl * z1 * b10
                + 2.0 * bl * sd * z1 * b16
            )
            a10 = rho * (
                2.0 * bl * z2 * b6
                + (bl_squared + z1 * z3 * sd_squared - z1 * z2) * b10
                + 2.0 * sd * z1 * z2 * b16
            )
            a16 = rho * (
                -2.0 * bl * sd * z3 * b6
                + 2.0 * sd * z1 * z3 * b10
                + (bl_squared - z1 * z3 * sd_squared + z1 * z2) * b16
            )

            # Abwaertspfad durch S1, S2, P1 und S3.
            electrical_sum = re + z5
            a5 = (
                -(z5 / electrical_sum) * b4
                + (re / electrical_sum) * b5
                - (z5 / electrical_sum) * a6
            )

            mechanical_sum = z7 + z8 + z9
            a7 = (
                ((z8 + z9) / mechanical_sum) * b7
                - (z7 / mechanical_sum) * b8
                - (z7 / mechanical_sum) * a10
            )
            a8 = (
                -(z8 / mechanical_sum) * b7
                + ((z7 + z9) / mechanical_sum) * b8
                - (z8 / mechanical_sum) * a10
            )
            a9 = (
                -(z9 / mechanical_sum) * b7
                - (z9 / mechanical_sum) * b8
                - (z9 / mechanical_sum) * a10
            )

            a13 = -(z13 / parallel_sum) * b13 + a16
            acoustic_sum = z11 + z12
            a12 = (
                (z11 / acoustic_sum) * b12
                - (z12 / acoustic_sum) * a13
            )

            current = (a5 - b5) / (2.0 * z5)
            ims = a9 / (2.0 * rms)
            pressure = 0.5 * (a16 + b16)
            x_new = xi_x * (ts * ims + x_prev1)
            volume_velocity = sd * ims

            output[index] = pressure
            if record_traces:
                displacement_trace[index] = x_new
                diaphragm_velocity_trace[index] = ims
                volume_velocity_trace[index] = volume_velocity
                voice_coil_current_trace[index] = current

            # Commit der fuer die naechste Abtastung benoetigten Zustaende.
            x_prev2 = x_prev1
            x_prev1 = x_new
            ims_prev = ims
            a5_prev = a5
            b5_prev = b5
            a7_prev = a7
            b7_prev = b7
            a8_prev = a8
            b8_prev = b8
            a12_prev = a12
            b12_prev = b12

            last_current = current
            last_pressure = pressure
            last_volume_velocity = volume_velocity
            last_displacement = x_new

        self.x_prev2 = x_prev2
        self.x_prev1 = x_prev1
        self.ims_prev = ims_prev
        self.a5_prev = a5_prev
        self.b5_prev = b5_prev
        self.a7_prev = a7_prev
        self.b7_prev = b7_prev
        self.a8_prev = a8_prev
        self.b8_prev = b8_prev
        self.a12_prev = a12_prev
        self.b12_prev = b12_prev
        self.z5 = z5
        self.z7 = z7 if sample_count else self.z7
        self.bl = bl
        self.last_current = last_current
        self.last_output_pressure = last_pressure
        self.last_volume_velocity = last_volume_velocity
        self.last_displacement = last_displacement

        if not record_traces:
            return output
        return {
            "input_voltage": input_array.copy(),
            "output_pressure": output,
            "displacement": displacement_trace,
            "diaphragm_velocity": diaphragm_velocity_trace,
            "volume_velocity": volume_velocity_trace,
            "voice_coil_current": voice_coil_current_trace,
        }

    def process_block_into(self, input_signal: np.ndarray, output_buffer: np.ndarray) -> np.ndarray:
        """Verarbeitet einen Block ohne einen neuen Ausgangsdatenpuffer anzulegen."""
        return self.process_block(input_signal, output_buffer=output_buffer)


def simulate_model(
    model: FastBernardiniSpeakerSolver,
    input_signal: np.ndarray,
    record_traces: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Kompatibler Blockeinsprung entsprechend ``WDFSolverVPres.simulate_model``."""
    if not isinstance(model, FastBernardiniSpeakerSolver):
        raise TypeError("WDFSolverFast.simulate_model erwartet FastBernardiniSpeakerSolver.")
    return model.process_block(input_signal, record_traces=record_traces)
