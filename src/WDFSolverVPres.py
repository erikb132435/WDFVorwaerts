
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import json
from pathlib import Path

# Grundlegende Hilfsfunktionen

def _safe_value(x, floor=1e-18):
    """Begrenzt bekannte nichtnegative Analysewerte auf eine positive Untergrenze."""
    return np.maximum(np.asarray(x, dtype=float), floor)


def _require_positive_scalar(value, name, minimum=0.0):
    """Validiert physikalisch positive Skalare, statt negative Werte still zu klemmen."""
    result = float(value)
    if not np.isfinite(result) or result <= float(minimum):
        raise ValueError(
            f"{name} muss groesser als {float(minimum):g} sein; erhalten: {result!r}."
        )
    return result

class PolynomialOrFunction:
    """Kapselt Kennlinien als Polynom oder frei uebergebene Funktion."""
    def __init__(self, coeffs=None, func=None, derivative=None):
        self.coeffs = None if coeffs is None else np.asarray(coeffs, dtype=float)
        self.func = func
        self.derivative = derivative

    def value(self, x):
        if self.coeffs is not None:
            return np.polyval(self.coeffs[::-1], x)
        if self.func is None:
            raise ValueError("Either coeffs or func must be provided.")
        return self.func(x)

    def deriv(self, x):
        if self.coeffs is not None:
            if len(self.coeffs) <= 1:
                return 0.0
            d = np.array([n * self.coeffs[n] for n in range(1, len(self.coeffs))], dtype=float)
            return np.polyval(d[::-1], x)
        if self.derivative is not None:
            return self.derivative(x)

        # Numerical fallback keeps the previous params-dict style usable.
        h = 1e-6 * max(1.0, abs(x))
        return (self.func(x + h) - self.func(x - h)) / (2.0 * h)

# WDF-Bausteine und Lautsprechermodell

class WDOnePort:
    """Basisklasse fuer WDF-Einports mit Portwiderstand und Reflexionszustand."""
    def __init__(self, port_id, resistance):
        self.port_id = port_id
        self.Z = _require_positive_scalar(resistance, f"Portwiderstand Z{port_id}")
        self.a_prev = 0.0
        self.b_prev = 0.0
        self.a = 0.0
        self.b = 0.0

    def reflect(self):
        self.b = self.b_prev
        return self.b

    def accept_incident(self, a):
        self.a = float(a)

    def commit(self):
        self.a_prev = self.a
        self.b_prev = self.b

class ResistiveVoltageSource(WDOnePort):
    """Modelliert eine spannungsgetriebene Quelle mit Serienwiderstand im WDF."""
    def __init__(self, port_id, resistance):
        super().__init__(port_id, resistance)
        self.vin = 0.0

    def set_input(self, vin):
        self.vin = float(vin)

    def reflect(self):
        self.b = self.vin
        return self.b

class ResistiveOnePort(WDOnePort):
    """Modelliert einen linearen ohmschen Einport ohne gespeicherten Zustand."""
    def reflect(self):
        self.b = 0.0
        return self.b

class MassOnePort(WDOnePort):
    """Modelliert die mechanische Masse als WDF-Einport."""
    def reflect(self):
        self.b = 0.5 * (self.b_prev - self.a_prev)
        return self.b

class AcousticComplianceOnePort(WDOnePort):
    """Modelliert die akustische Nachgiebigkeit des Gehaeuses als WDF-Einport."""
    def reflect(self):
        self.b = 0.5 * (self.b_prev + self.a_prev)
        return self.b

class NonlinearLeOnePort(WDOnePort):
    """Modelliert die auslenkungsabhaengige Schwingspuleninduktivitaet im WDF."""
    def __init__(self, port_id, fs, le_model, xi_le):
        super().__init__(port_id, 1.0)
        self.fs = fs
        self.ts = 1.0 / fs
        self.le_model = le_model
        self.xi_le = xi_le
        self.Le = 0.0
        self.Le_prime = 0.0
        self.ims_hat = 0.0
        self.Z_prev = 1.0

    def update_nonlinearity(self, x_hat, ims_hat):
        self.Z_prev = _require_positive_scalar(self.Z, "vorheriger Le-Portwiderstand", 1e-9)
        self.Le = _require_positive_scalar(self.le_model.value(x_hat), "Le(x)", 1e-9)
        self.Le_prime = float(self.le_model.deriv(x_hat))
        self.ims_hat = float(ims_hat)
        self.Z = _require_positive_scalar(
            self.Le_prime * self.ims_hat + self.Le / self.ts,
            "zeitvarianter Le-Portwiderstand",
            1e-9,
        )

    def reflect(self):
        z_prev = _require_positive_scalar(self.Z_prev, "vorheriger Le-Portwiderstand", 1e-9)
        self.b = self.xi_le * (self.b_prev - self.a_prev) * self.Le / (2.0 * self.ts * z_prev)
        return self.b

class NonlinearKmsOnePort(WDOnePort):
    """Modelliert die auslenkungsabhaengige mechanische Steifigkeit im WDF."""
    def __init__(self, port_id, fs, kms_model, xi_cms):
        super().__init__(port_id, 1.0)
        self.fs = fs
        self.ts = 1.0 / fs
        self.kms_model = kms_model
        self.xi_cms = xi_cms
        self.Kms = 0.0
        self.Kms_prime = 0.0
        self.Cms_prime = 0.0

    def update_nonlinearity(self, x_hat):
        self.Kms = _require_positive_scalar(self.kms_model.value(x_hat), "Kms(x)", 1e-9)
        self.Kms_prime = float(self.kms_model.deriv(x_hat))
        self.Cms_prime = -self.Kms_prime / (self.Kms ** 2)
        v_prev = 0.5 * (self.a_prev + self.b_prev)
        factor = 1.0 - self.Cms_prime * v_prev
        self.Z = _require_positive_scalar(
            self.Kms * self.ts * factor,
            "zeitvarianter Kms-Portwiderstand",
            1e-9,
        )

    def reflect(self):
        self.b = 0.5 * self.xi_cms * (self.a_prev + self.b_prev)
        return self.b

class SeriesAdaptor:
    """Koppelt zwei WDF-Ports in Serie."""
    def __init__(self, name, child_ports, parent_port):
        self.name = name
        self.child_ports = list(child_ports)
        self.parent_port = parent_port

    def scatter_parent(self, child_incidents, child_resistances):
        raise NotImplementedError

    def scatter_children(self, child_incidents, child_resistances, parent_incident):
        raise NotImplementedError

class SeriesAdaptor3(SeriesAdaptor):
    """Koppelt drei WDF-Ports in Serie."""
    def scatter_parent(self, child_incidents, child_resistances):
        a1, a2 = map(float, child_incidents)
        return -a1 - a2

    def scatter_children(self, child_incidents, child_resistances, parent_incident):
        a1, a2 = map(float, child_incidents)
        z1, z2 = map(float, child_resistances)
        zsum = _safe_value(z1 + z2, 1e-12)
        b1 = (z2 / zsum) * a1 - (z1 / zsum) * a2 - (z1 / zsum) * parent_incident
        b2 = -(z2 / zsum) * a1 + (z1 / zsum) * a2 - (z2 / zsum) * parent_incident
        return np.array([b1, b2], dtype=float)

class SeriesAdaptor4(SeriesAdaptor):
    """Koppelt vier WDF-Ports in Serie."""
    def scatter_parent(self, child_incidents, child_resistances):
        a1, a2, a3 = map(float, child_incidents)
        return -a1 - a2 - a3

    def scatter_children(self, child_incidents, child_resistances, parent_incident):
        a1, a2, a3 = map(float, child_incidents)
        z1, z2, z3 = map(float, child_resistances)
        zsum = _safe_value(z1 + z2 + z3, 1e-12)
        b1 = ((z2 + z3) / zsum) * a1 - (z1 / zsum) * a2 - (z1 / zsum) * a3 - (z1 / zsum) * parent_incident
        b2 = -(z2 / zsum) * a1 + ((z1 + z3) / zsum) * a2 - (z2 / zsum) * a3 - (z2 / zsum) * parent_incident
        b3 = -(z3 / zsum) * a1 - (z3 / zsum) * a2 + ((z1 + z2) / zsum) * a3 - (z3 / zsum) * parent_incident
        return np.array([b1, b2, b3], dtype=float)

class ParallelAdaptor3:
    """Koppelt drei WDF-Ports parallel."""
    def __init__(self, name, child_ports, parent_port):
        self.name = name
        self.child_ports = list(child_ports)
        self.parent_port = parent_port

    def equivalent_resistance(self, child_resistances):
        z1, z2 = map(float, child_resistances)
        return float(_safe_value((z1 * z2) / _safe_value(z1 + z2, 1e-12), 1e-12))

    def scatter_parent(self, child_incidents, child_resistances):
        a1, a2 = map(float, child_incidents)
        z1, z2 = map(float, child_resistances)
        zsum = _safe_value(z1 + z2, 1e-12)
        return (z2 / zsum) * a1 + (z1 / zsum) * a2

    def scatter_children(self, child_incidents, child_resistances, parent_incident):
        a1, a2 = map(float, child_incidents)
        z1, z2 = map(float, child_resistances)
        zsum = _safe_value(z1 + z2, 1e-12)
        b1 = -(z1 / zsum) * a1 + (z1 / zsum) * a2 + parent_incident
        b2 = (z2 / zsum) * a1 - (z2 / zsum) * a2 + parent_incident
        return np.array([b1, b2], dtype=float)

class NonReciprocalRoot:
    """Realisiert die nichtreziproke Root-Kopplung des Bernardini-Modells."""
    def __init__(self):
        self.a = np.zeros(3, dtype=float)
        self.b = np.zeros(3, dtype=float)

    def scatter(self, a_vec, z1, z2, z3, bl, sd):
        a1, a2, a3 = map(float, a_vec)
        rho = 1.0 / _safe_value(bl * bl + z1 * z3 * sd * sd + z1 * z2, 1e-12)
        S = rho * np.array(
            [
                [bl * bl - z1 * z3 * sd * sd - z1 * z2, -2.0 * bl * z1, 2.0 * bl * sd * z1],
                [2.0 * bl * z2, bl * bl + z1 * z3 * sd * sd - z1 * z2, 2.0 * sd * z1 * z2],
                [-2.0 * bl * sd * z3, 2.0 * sd * z1 * z3, bl * bl - z1 * z3 * sd * sd + z1 * z2],
            ],
            dtype=float,
        )
        self.a[:] = [a1, a2, a3]
        self.b = S @ self.a
        return self.b.copy()

class BernardiniSpeakerSolver:
    """Fuehrt einen Simulationsschritt des Bernardini-Forward-Modells aus."""
    def __init__(self, fs, params):
        self.fs = float(fs)
        self.ts = 1.0 / self.fs

        self.Re = _require_positive_scalar(params["Re"], "Re")
        self.Mms = _require_positive_scalar(params["Mms"], "Mms")
        self.Rms = _require_positive_scalar(params["Rms"], "Rms")
        self.Sd = _require_positive_scalar(params["Sd"], "Sd")
        self.Rcab = _require_positive_scalar(params.get("Rcab", 1e6), "Rcab")
        self.Ccab = _require_positive_scalar(params.get("Ccab", 1e-12), "Ccab")
        self.Ral = _require_positive_scalar(params.get("Ral", 1e6), "Ral")

        self.xi_x = float(params.get("xi_x", 0.9999))
        self.xi_le = float(params.get("xi_Le", 0.9999))
        self.xi_cms = float(params.get("xi_Cms", 0.9999))

        self.bl_model = self._build_model(params, "Bl")
        self.le_model = self._build_model(params, "Le")
        self.kms_model = self._build_model(params, "Kms")

        self.x_prev2 = 0.0
        self.x_prev1 = 0.0
        self.ims_prev = 0.0
        self.bl = float(self.bl_model.value(0.0))

        self.source = ResistiveVoltageSource(4, self.Re)
        self.le = NonlinearLeOnePort(5, self.fs, self.le_model, self.xi_le)
        self.kms = NonlinearKmsOnePort(7, self.fs, self.kms_model, self.xi_cms)
        self.mass = MassOnePort(8, self.Mms * self.fs)
        self.rms = ResistiveOnePort(9, self.Rms)
        self.rcab = ResistiveOnePort(11, self.Rcab)
        self.ccab = AcousticComplianceOnePort(12, self.ts / self.Ccab)
        self.ral = ResistiveOnePort(15, self.Ral)

        self.S1 = SeriesAdaptor3("S1", [4, 5], 6)
        self.S2 = SeriesAdaptor4("S2", [7, 8, 9], 10)
        self.S3 = SeriesAdaptor3("S3", [11, 12], 13)
        self.P1 = ParallelAdaptor3("P1", [14, 15], 16)
        self.root = NonReciprocalRoot()

        self.last_output_pressure = 0.0
        self.last_volume_velocity = 0.0
        self.last_displacement = 0.0
        self.last_current = 0.0

    def _build_model(self, params, stem):
        coeff_key = f"{stem}_coeffs"
        func_key = f"{stem}_func"
        deriv_key = f"{stem}_deriv"
        coeffs = params.get(coeff_key)
        func = params.get(func_key)
        deriv = params.get(deriv_key)
        if coeffs is None and func is None:
            raise ValueError(f"Missing {stem} model. Use `{coeff_key}` or `{func_key}`.")
        return PolynomialOrFunction(coeffs=coeffs, func=func, derivative=deriv)

    def _commit_ports(self):
        for port in [self.source, self.le, self.kms, self.mass, self.rms, self.rcab, self.ccab, self.ral]:
            port.commit()

    def step(self, vin):
        self.source.set_input(vin)

        x_hat = self.x_prev1
        ims_hat = self.ims_prev

        self.bl = float(self.bl_model.value(x_hat))
        self.le.update_nonlinearity(x_hat, ims_hat)
        self.kms.update_nonlinearity(x_hat)

        # Upward pass: adapted one-ports reflect into the tree.
        b4 = self.source.reflect()
        b5 = self.le.reflect()
        b7 = self.kms.reflect()
        b8 = self.mass.reflect()
        b9 = self.rms.reflect()
        b11 = self.rcab.reflect()
        b12 = self.ccab.reflect()
        b15 = self.ral.reflect()

        b6 = self.S1.scatter_parent([b4, b5], [self.source.Z, self.le.Z])
        b10 = self.S2.scatter_parent([b7, b8, b9], [self.kms.Z, self.mass.Z, self.rms.Z])
        z13 = self.kms.Z + self.mass.Z  # temporary placeholder overwritten below
        b13 = self.S3.scatter_parent([b11, b12], [self.rcab.Z, self.ccab.Z])
        z14 = self.rcab.Z + self.ccab.Z
        b16 = self.P1.scatter_parent([b13, b15], [z14, self.ral.Z])

        z10 = self.kms.Z + self.mass.Z + self.rms.Z
        z13 = self.rcab.Z + self.ccab.Z
        z16 = self.P1.equivalent_resistance([z13, self.ral.Z])

        # Root scattering: Bernardini non-reciprocal 3-port.
        root_b = self.root.scatter(
            [b6, b10, b16],
            self.Re + self.le.Z,
            z10,
            z16,
            self.bl,
            self.Sd,
        )
        a6, a10, a16 = root_b

        # Downward pass: junctions send incidents to the one-ports.
        a4, a5 = self.S1.scatter_children([b4, b5], [self.source.Z, self.le.Z], a6)
        a7, a8, a9 = self.S2.scatter_children([b7, b8, b9], [self.kms.Z, self.mass.Z, self.rms.Z], a10)
        a13, a15 = self.P1.scatter_children([b13, b15], [z13, self.ral.Z], a16)
        a11, a12 = self.S3.scatter_children([b11, b12], [self.rcab.Z, self.ccab.Z], a13)

        self.source.accept_incident(a4)
        self.le.accept_incident(a5)
        self.kms.accept_incident(a7)
        self.mass.accept_incident(a8)
        self.rms.accept_incident(a9)
        self.rcab.accept_incident(a11)
        self.ccab.accept_incident(a12)
        self.ral.accept_incident(a15)

        # Branch variables.
        i_e = (a5 - b5) / (
            2.0 * _require_positive_scalar(self.le.Z, "Le-Portwiderstand", 1e-12)
        )
        i_ms = (a9 - b9) / (
            2.0 * _require_positive_scalar(self.rms.Z, "Rms-Portwiderstand", 1e-12)
        )
        p_out = 0.5 * (a16 + b16)

        x_new = self.xi_x * (self.ts * i_ms + self.x_prev1)
        self.x_prev2 = self.x_prev1
        self.x_prev1 = x_new
        self.ims_prev = i_ms

        self.last_output_pressure = p_out
        self.last_volume_velocity = self.Sd * i_ms
        self.last_displacement = x_new
        self.last_current = i_e

        self._commit_ports()
        return p_out

# Presets und Koeffizientenumrechnungen

def _convert_bl_coeffs_from_mm(coeffs):
    """Wandelt die Paper-Koeffizienten von Bl(x) in SI-Einheiten um."""
    # Paper Table 4: Bl_n in N / (A * mm^n)
    return [float(c) * (1000.0 ** n) for n, c in enumerate(coeffs)]

def _convert_le_coeffs_from_mm_mh(coeffs):
    """Wandelt die Paper-Koeffizienten von Le(x) in SI-Einheiten um."""
    # Paper Table 4: Le_n in mH / mm^n
    return [float(c) * 1e-3 * (1000.0 ** n) for n, c in enumerate(coeffs)]

def _convert_kms_coeffs_from_mm(coeffs):
    """Wandelt die Paper-Koeffizienten von Kms(x) in SI-Einheiten um."""
    # Paper Table 4: Kms_n in N / mm^(n+1), convert to N / m^(n+1)
    return [float(c) * (1000.0 ** (n + 1)) for n, c in enumerate(coeffs)]

def make_bernardini_preset(name):
    """Erzeugt ein Lautsprecher-Preset aus den Bernardini-Tabellenwerten."""
    # Values from Bernardini et al., Part I, Table 2 and Table 4.
    presets_raw = {
        "Spk-1": {
            "Re_ohm": 5.91,
            "Mms_g": 38.606,
            "Rms": 2.814,
            "Sd_cm2": 539.13,
            "resonance_hz": 57.5,
            "Rcab": 18.7072,
            "Ccab": 0.0000071487,
            "Ral": 3741.4,
            "Bl_coeffs_mm": [13.854, -0.0077114, -0.6019057, -0.0016604, 0.0101072],
            "Le_coeffs_mm_mh": [0.547, -0.0478768, -0.0113542, 0.0010058, 0.000113],
            "Kms_coeffs_mm": [4.99, -0.2607102, 0.1466057, -0.0049665, 0.0005919],
        },
        "Spk-2": {
            "Re_ohm": 3.29,
            "Mms_g": 1.476,
            "Rms": 0.344,
            "Sd_cm2": 17.87,
            "resonance_hz": 162.8,
            "Rcab": 136.9273,
            "Ccab": 0.0000071487,
            "Ral": 1369.3,
            "Bl_coeffs_mm": [2.219, 0.15563, -0.1576, -0.0024596, 0.0033673],
            "Le_coeffs_mm_mh": [0.036, -0.016303, 0.0018228, 0.00072416, -0.000060094],
            "Kms_coeffs_mm": [1.54, -0.34898, 0.10891, -0.0090764, 0.0028738],
        },
    }

    raw = presets_raw[name]
    return {
        "preset_name": name,
        "Re": raw["Re_ohm"],
        "Mms": raw["Mms_g"] * 1e-3,
        "Rms": raw["Rms"],
        "Sd": raw["Sd_cm2"] * 1e-4,
        "resonance_hz": raw["resonance_hz"],
        "Rcab": raw["Rcab"],
        "Ccab": raw["Ccab"],
        "Ral": raw["Ral"],
        "Bl_coeffs": _convert_bl_coeffs_from_mm(raw["Bl_coeffs_mm"]),
        "Le_coeffs": _convert_le_coeffs_from_mm_mh(raw["Le_coeffs_mm_mh"]),
        "Kms_coeffs": _convert_kms_coeffs_from_mm(raw["Kms_coeffs_mm"]),
        "xi_x": 0.9999,
        "xi_Le": 0.9999,
        "xi_Cms": 0.9999,
    }

BERNARDINI_PRESETS = {
    "Spk-1": make_bernardini_preset("Spk-1"),
    "Spk-2": make_bernardini_preset("Spk-2"),
}

# Signalerzeugung und Simulation

def simulate_model(model, input_signal, record_traces=False):
    """Berechnet das Modell fuer ein komplettes Eingangssignal.

    Ohne ``record_traces`` bleibt Rueckgabewert und Rechenpfad kompatibel zum
    bisherigen Solver: Zurueckgegeben wird ausschliesslich ``Pout``. Im
    optionalen Analysemodus werden bereits berechnete interne Modellgroessen
    sampleweise aufgezeichnet und gemeinsam in einem Dictionary geliefert.
    """
    output = np.zeros_like(input_signal, dtype=float)
    if not record_traces:
        for n, sample in enumerate(input_signal):
            output[n] = model.step(sample)
        return output

    displacement = np.zeros_like(input_signal, dtype=float)
    diaphragm_velocity = np.zeros_like(input_signal, dtype=float)
    volume_velocity = np.zeros_like(input_signal, dtype=float)
    voice_coil_current = np.zeros_like(input_signal, dtype=float)
    for n, sample in enumerate(input_signal):
        output[n] = model.step(sample)
        displacement[n] = model.last_displacement
        diaphragm_velocity[n] = model.ims_prev
        volume_velocity[n] = model.last_volume_velocity
        voice_coil_current[n] = model.last_current
    return {
        "input_voltage": np.asarray(input_signal, dtype=float),
        "output_pressure": output,
        "displacement": displacement,
        "diaphragm_velocity": diaphragm_velocity,
        "volume_velocity": volume_velocity,
        "voice_coil_current": voice_coil_current,
    }

def generate_sine(fs, duration_s, freq_hz, vrms):
    """Erzeugt ein Sinussignal mit vorgegebenem RMS-Pegel."""
    t = np.arange(int(fs * duration_s)) / fs
    amplitude = vrms * np.sqrt(2.0)
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

def generate_two_tone(
    fs,
    duration_s,
    freq1_hz,
    freq2_hz,
    vrms,
    phase1_rad=0.0,
    phase2_rad=0.0,
):
    """Erzeugt ein normiertes Two-Tone-Signal nach Bernardini mit Phasenparametern."""
    t = np.arange(int(fs * duration_s)) / fs
    signal = (
        np.sin(2.0 * np.pi * freq1_hz * t + phase1_rad)
        + np.sin(2.0 * np.pi * freq2_hz * t + phase2_rad)
    )
    signal_rms = float(np.sqrt(np.mean(signal ** 2)))
    return float(vrms) * signal / _safe_value(signal_rms, 1e-18)

def generate_sweep(fs, T, f1, f2, Vrms):
    """Erzeugt einen exponentiellen Sweep samt Inversfilter."""
    N = int(fs * T)
    t = np.arange(N) / fs
    L = np.log(f2 / f1)

    phase = 2.0 * np.pi * f1 * T / L * (np.exp(t * L / T) - 1.0)
    sweep = np.sin(phase)
    sweep *= Vrms / np.sqrt(np.mean(sweep ** 2))

    k = np.exp(t * L / T)
    inv = sweep[::-1] / k
    return sweep, inv

def make_verification_sine(fs, params, vrms, duration_s=1.0):
    """Erzeugt den Fig.-6-aehnlichen Verifikationssinus nahe halber Resonanz."""
    # Vergleichbar zur Verifikation in Bernardini Fig. 6:
    # Ein stationaerer Sinus mit fin ungefaehr bei fr / 2.
    resonance_hz = estimate_small_signal_resonance(params)
    freq_hz = 0.5 * resonance_hz
    tone = generate_sine(fs, duration_s, freq_hz, vrms)
    return tone, freq_hz, resonance_hz

# Signalaufbereitung und Spektralanalyse

def _rounded_gcd(values):
    """Bestimmt einen ganzzahligen groessten gemeinsamen Teiler fuer Frequenzhilfen."""
    values = [int(abs(v)) for v in values if int(abs(v)) > 0]
    if not values:
        return 0
    gcd_value = values[0]
    for value in values[1:]:
        gcd_value = int(np.gcd(gcd_value, value))
    return gcd_value

def prepare_analysis_signal(signal, fs, settle_s=0.0, tone_freqs_hz=None):
    """Schneidet das stationaere Signal frei und entfernt den Gleichanteil."""
    start = int(max(settle_s, 0.0) * fs)
    analysis_signal = np.asarray(signal[start:], dtype=float)
    if analysis_signal.size == 0:
        return analysis_signal

    if tone_freqs_hz:
        scale = 1000.0
        gcd_millihz = _rounded_gcd([round(freq * scale) for freq in tone_freqs_hz if freq and freq > 0.0])
        if gcd_millihz > 0:
            fundamental_hz = gcd_millihz / scale
            samples_per_period = fs / fundamental_hz
            num_periods = int(len(analysis_signal) / samples_per_period)
            if num_periods >= 2:
                trimmed_length = int(num_periods * samples_per_period)
                analysis_signal = analysis_signal[:trimmed_length]

    return analysis_signal - np.mean(analysis_signal)

def prepare_two_tone_analysis_signal(signal, fs, settle_s=0.0):
    """Bereitet ein Two-Tone-Ausgangssignal fuer die Spektralanalyse auf."""
    start = int(max(settle_s, 0.0) * fs)
    analysis_signal = np.asarray(signal[start:], dtype=float)
    if analysis_signal.size == 0:
        return analysis_signal
    return analysis_signal - np.mean(analysis_signal)

def _analysis_window(n_samples):
    """Liefert das Analysefenster und dessen koharente Verstaerkung."""
    window = np.hanning(int(n_samples))
    coherent_gain = np.sum(window) / max(len(window), 1)
    return window, float(_safe_value(coherent_gain, 1e-12))

def compute_stationary_spectrum(signal, fs):
    """Berechnet Linienamplituden und mittlere Leistung stationaerer Komponenten.

    Positive Frequenzen werden einseitig skaliert. ``magnitude`` entspricht
    dabei der Spitzenamplitude eines kohaerenten Sinustons, ``power`` dessen
    mittlerem Quadrat (A_peak**2 / 2). DC und ggf. Nyquist werden nicht
    verdoppelt.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < 2:
        raise ValueError("Fuer ein stationaeres Spektrum werden mindestens zwei Samples benoetigt.")
    window, coherent_gain = _analysis_window(len(signal))
    spectrum = np.fft.rfft(signal * window)
    magnitude = np.abs(spectrum) / (len(signal) * coherent_gain)

    if len(signal) % 2 == 0:
        positive_frequency_slice = slice(1, -1)
    else:
        positive_frequency_slice = slice(1, None)
    magnitude[positive_frequency_slice] *= 2.0

    power = magnitude ** 2
    power[positive_frequency_slice] *= 0.5
    freqs = np.fft.rfftfreq(len(signal), 1.0 / fs)
    power_db = 10.0 * np.log10(_safe_value(power, 1e-24))
    return freqs, magnitude, power_db

def compute_psd_db(signal, fs, nperseg=8192):
    """Berechnet die PSD eines Signals in dB pro Hertz."""
    signal = np.asarray(signal, dtype=float)
    nperseg = min(int(nperseg), len(signal))
    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
        detrend=False,
    )
    return freqs, 10.0 * np.log10(_safe_value(psd, 1e-24))

def compute_highres_psd_db(signal, fs, zero_pad_factor=8):
    """Berechnet eine hochaufgeloeste PSD fuer figurnahe Spektralvergleiche."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    nperseg = len(signal)
    base_nfft = 1 << int(np.ceil(np.log2(max(nperseg, 2))))
    nfft = max(nperseg, int(base_nfft * max(int(zero_pad_factor), 1)))
    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=0,
        nfft=nfft,
        scaling="density",
        detrend=False,
    )
    return freqs, 10.0 * np.log10(_safe_value(psd, 1e-24))

def compute_power_spectrum_db(signal, fs):
    """Berechnet das Leistungsspektrum eines Signals in dB."""
    freqs, _, power_db = compute_stationary_spectrum(signal, fs)
    return freqs, power_db

def stationary_tone_timing(freq_hz, settle_periods=8, measure_periods=16, min_total_s=0.5):
    """Bestimmt Einschwing- und Messdauer fuer stationaere Sinusmessungen."""
    freq_safe = float(_safe_value(freq_hz, 1e-12))
    settle_s = settle_periods / freq_safe
    measure_s = measure_periods / freq_safe
    total_s = max(min_total_s, settle_s + measure_s)
    return total_s, settle_s

def measure_harmonic_levels_db(signal, fs, fundamental_hz, harmonics=8):
    """Misst Harmonische eines stationaeren Signals relativ zum Grundton."""
    analysis_signal = prepare_analysis_signal(signal, fs, settle_s=0.0, tone_freqs_hz=[fundamental_hz])
    freqs, magnitude, power_db = compute_stationary_spectrum(analysis_signal, fs)
    maximum_harmonic = int(np.floor((fs / 2.0) / float(_safe_value(fundamental_hz, 1e-18))))
    evaluated_harmonics = min(int(harmonics), maximum_harmonic)
    if evaluated_harmonics < 1:
        raise ValueError("Der Grundton liegt nicht unterhalb der Nyquist-Frequenz.")
    measurements = []
    for k in range(1, evaluated_harmonics + 1):
        target = k * fundamental_hz
        idx = int(np.argmin(np.abs(freqs - target)))
        measurements.append(
            {
                "harmonic": k,
                "target_hz": float(target),
                "bin_hz": float(freqs[idx]),
                "magnitude": float(magnitude[idx]),
                "power_db": float(power_db[idx]),
            }
        )
    ref_mag = float(_safe_value(measurements[0]["magnitude"], 1e-18))
    for item in measurements:
        item["relative_db"] = 20.0 * np.log10(float(_safe_value(item["magnitude"], 1e-18)) / ref_mag)
    return measurements, freqs, power_db

def compute_stationary_sine_thd_point(
    fs,
    params,
    vrms,
    freq_hz,
    harmonics=5,
    settle_periods=8,
    measure_periods=16,
    min_total_s=0.5,
):
    """Berechnet einen einzelnen THD-Messpunkt ueber stationaere Sinusanregung."""
    duration_s, settle_s = stationary_tone_timing(
        freq_hz,
        settle_periods=settle_periods,
        measure_periods=measure_periods,
        min_total_s=min_total_s,
    )
    tone = generate_sine(fs, duration_s, freq_hz, vrms)
    model = BernardiniSpeakerSolver(fs, params)
    output = simulate_model(model, tone)
    analysis_signal = prepare_analysis_signal(output, fs, settle_s=settle_s, tone_freqs_hz=[freq_hz])
    harmonic_data, _, _ = measure_harmonic_levels_db(analysis_signal, fs, freq_hz, harmonics=harmonics)

    h1_mag = float(_safe_value(harmonic_data[0]["magnitude"], 1e-18))
    squared_ratio_sum = 0.0
    dominant_harmonic = None
    dominant_relative_db = -np.inf
    for item in harmonic_data[1:]:
        ratio = float(item["magnitude"]) / h1_mag
        squared_ratio_sum += ratio * ratio
        if item["relative_db"] > dominant_relative_db:
            dominant_relative_db = float(item["relative_db"])
            dominant_harmonic = int(item["harmonic"])

    thd = np.sqrt(squared_ratio_sum)
    thd_db = 20.0 * np.log10(float(_safe_value(thd, 1e-18)))
    return {
        "freq_hz": float(freq_hz),
        "thd_db": float(thd_db),
        "harmonic_data": harmonic_data,
        "dominant_harmonic": dominant_harmonic,
        "dominant_relative_db": float(dominant_relative_db),
        "duration_s": float(duration_s),
        "settle_s": float(settle_s),
    }

def compute_output_thd_point(
    output_signal,
    fs,
    freq_hz,
    harmonics=5,
    settle_s=0.0,
):
    """Berechnet den THD eines bereits vorliegenden Ausgangssignals."""
    analysis_signal = prepare_analysis_signal(
        output_signal,
        fs,
        settle_s=settle_s,
        tone_freqs_hz=[freq_hz],
    )
    harmonic_data, _, _ = measure_harmonic_levels_db(analysis_signal, fs, freq_hz, harmonics=harmonics)

    h1_mag = float(_safe_value(harmonic_data[0]["magnitude"], 1e-18))
    squared_ratio_sum = 0.0
    dominant_harmonic = None
    dominant_relative_db = -np.inf
    for item in harmonic_data[1:]:
        ratio = float(item["magnitude"]) / h1_mag
        squared_ratio_sum += ratio * ratio
        if item["relative_db"] > dominant_relative_db:
            dominant_relative_db = float(item["relative_db"])
            dominant_harmonic = int(item["harmonic"])

    thd = np.sqrt(squared_ratio_sum)
    thd_db = 20.0 * np.log10(float(_safe_value(thd, 1e-18)))
    return {
        "freq_hz": float(freq_hz),
        "thd_db": float(thd_db),
        "harmonic_data": harmonic_data,
        "dominant_harmonic": dominant_harmonic,
        "dominant_relative_db": float(dominant_relative_db),
        "settle_s": float(settle_s),
        "analysis_signal": analysis_signal,
    }

def build_log_frequency_grid(f1_hz, f2_hz, num_points, include_freqs_hz=None):
    """Erzeugt ein logarithmisches Frequenzraster fuer Sweeps ueber Einzeltoene."""
    if num_points <= 1:
        grid = np.asarray([float(f1_hz)], dtype=float)
    else:
        grid = np.logspace(np.log10(float(f1_hz)), np.log10(float(f2_hz)), int(num_points))

    if include_freqs_hz:
        extras = [float(freq) for freq in include_freqs_hz if float(f1_hz) <= float(freq) <= float(f2_hz)]
        if extras:
            grid = np.unique(np.concatenate([grid, np.asarray(extras, dtype=float)]))
            grid.sort()

    return grid

def compute_stationary_sine_thd_curve(
    fs,
    params,
    vrms,
    f1_hz,
    f2_hz,
    num_points=24,
    harmonics=5,
    settle_periods=8,
    measure_periods=16,
    min_total_s=0.5,
    include_freqs_hz=None,
):
    """Berechnet eine THD-Kurve aus vielen stationaeren Sinusmessungen."""
    freqs = build_log_frequency_grid(f1_hz, f2_hz, num_points, include_freqs_hz=include_freqs_hz)
    thd_db = np.full(len(freqs), np.nan, dtype=float)
    dominant_harmonics = np.full(len(freqs), np.nan, dtype=float)
    dominant_relative_db = np.full(len(freqs), np.nan, dtype=float)

    for idx, freq_hz in enumerate(freqs):
        point = compute_stationary_sine_thd_point(
            fs,
            params,
            vrms,
            float(freq_hz),
            harmonics=harmonics,
            settle_periods=settle_periods,
            measure_periods=measure_periods,
            min_total_s=min_total_s,
        )
        thd_db[idx] = point["thd_db"]
        dominant_harmonics[idx] = point["dominant_harmonic"] if point["dominant_harmonic"] is not None else np.nan
        dominant_relative_db[idx] = point["dominant_relative_db"]

    return freqs, thd_db, dominant_harmonics, dominant_relative_db

# Hilfsfunktionen fuer papernahe Standardparameter

def _linear_input_impedance_magnitude(freq_hz, params):
    """Betrag der kontinuierlichen Kleinsignal-Eingangsimpedanz des Ersatzmodells."""
    frequency = _require_positive_scalar(freq_hz, "Frequenz")
    angular_frequency = 2.0 * np.pi * frequency
    s = 1j * angular_frequency
    kms0 = params["Kms_coeffs"][0] if "Kms_coeffs" in params else params["Kms_func"](0.0)
    le0 = params["Le_coeffs"][0] if "Le_coeffs" in params else params["Le_func"](0.0)
    bl0 = params["Bl_coeffs"][0] if "Bl_coeffs" in params else params["Bl_func"](0.0)
    kms0 = _require_positive_scalar(kms0, "Kms(0)")
    le0 = _require_positive_scalar(le0, "Le(0)")
    mms = _require_positive_scalar(params["Mms"], "Mms")
    rms = _require_positive_scalar(params["Rms"], "Rms")
    re = _require_positive_scalar(params["Re"], "Re")
    sd = _require_positive_scalar(params["Sd"], "Sd")
    rcab = _require_positive_scalar(params.get("Rcab", 1e6), "Rcab")
    ccab = _require_positive_scalar(params.get("Ccab", 1e-12), "Ccab")
    ral = _require_positive_scalar(params.get("Ral", 1e6), "Ral")

    cabinet_branch = rcab + 1.0 / (s * ccab)
    acoustic_load = cabinet_branch * ral / (cabinet_branch + ral)
    mechanical_impedance = rms + s * mms + kms0 / s + sd * sd * acoustic_load
    electric_impedance = re + s * le0 + float(bl0) ** 2 / mechanical_impedance
    return float(abs(electric_impedance))


def estimate_small_signal_resonance(params):
    """Liefert das gemessene Paper-Resonanzmaximum oder bestimmt max |Ze(f)|."""
    if "resonance_hz" in params:
        return _require_positive_scalar(params["resonance_hz"], "Resonanzfrequenz")

    kms0 = params["Kms_coeffs"][0] if "Kms_coeffs" in params else params["Kms_func"](0.0)
    uncoupled_hz = np.sqrt(
        _require_positive_scalar(kms0, "Kms(0)")
        / _require_positive_scalar(params["Mms"], "Mms")
    ) / (2.0 * np.pi)
    lower_hz = max(1e-3, 0.25 * uncoupled_hz)
    upper_hz = 4.0 * uncoupled_hz
    # Goldener-Schnitt-Suche auf dem physikalisch relevanten Intervall.
    ratio = 0.5 * (np.sqrt(5.0) - 1.0)
    left = lower_hz
    right = upper_hz
    inner_left = right - ratio * (right - left)
    inner_right = left + ratio * (right - left)
    value_left = _linear_input_impedance_magnitude(inner_left, params)
    value_right = _linear_input_impedance_magnitude(inner_right, params)
    for _ in range(96):
        if value_left < value_right:
            left = inner_left
            inner_left = inner_right
            value_left = value_right
            inner_right = left + ratio * (right - left)
            value_right = _linear_input_impedance_magnitude(inner_right, params)
        else:
            right = inner_right
            inner_right = inner_left
            value_right = value_left
            inner_left = right - ratio * (right - left)
            value_left = _linear_input_impedance_magnitude(inner_left, params)
    return float(0.5 * (left + right))

def get_paper_two_tone_defaults(preset_name):
    """Liefert die Fig.-7-aehnlichen Standardwerte fuer Two-Tone-Faelle."""
    defaults = {
        "Spk-1": {
            "vrms": 11.5,
            "freq1_hz": 28.75,
            "freq2_hz": 121.78,
            "phase1_rad": 0.8472,
            "phase2_rad": 2.2179,
        },
        "Spk-2": {
            "vrms": 3.0,
            "freq1_hz": 81.4,
            "freq2_hz": 344.79,
            "phase1_rad": 0.8472,
            "phase2_rad": 2.2179,
        },
    }
    return dict(defaults.get(preset_name, defaults["Spk-1"]))

def get_fig5_displacement_defaults_mm(preset_name):
    """Liefert presetabhaengige Standardbereiche fuer Fig.-5-Kennlinien."""
    defaults = {
        "Spk-1": (-5.0, 5.0),
        "Spk-2": (-2.5, 2.5),
    }
    return defaults.get(preset_name, (-5.0, 5.0))

# Plot- und Auswertefunktionen

def plot_nonlinearities_vs_displacement(params, preset_name, x_min_mm, x_max_mm, num_points=801):
    """Stellt Bl(x), Le(x) und Kms(x) fig.-5-aehnlich ueber x dar."""
    x_mm = np.linspace(float(x_min_mm), float(x_max_mm), int(num_points))
    x_m = x_mm * 1e-3

    bl_curve = np.polyval(np.asarray(params["Bl_coeffs"], dtype=float)[::-1], x_m)
    le_curve_mh = 1e3 * np.polyval(np.asarray(params["Le_coeffs"], dtype=float)[::-1], x_m)
    kms_curve_n_per_mm = 1e-3 * np.polyval(np.asarray(params["Kms_coeffs"], dtype=float)[::-1], x_m)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(x_mm, bl_curve, color="tab:blue", lw=1.6, label=r"$Bl(x)$")
    ax1.plot(x_mm, kms_curve_n_per_mm, color="tab:blue", lw=1.2, ls="--", label=r"$K_{ms}(x)$")
    ax2.plot(x_mm, le_curve_mh, color="tab:orange", lw=1.2, label=r"$L_e(x)$")

    ax1.set_title(f"Nichtlineare Kennlinien ueber x ({preset_name}, Fig.-5-artig)")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel(r"$Bl(x)$ [N/A], $K_{ms}(x)$ [N/mm]", color="tab:blue")
    ax2.set_ylabel(r"$L_e(x)$ [mH]", color="tab:orange")
    ax1.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:orange")
    ax1.grid(True, which="both", ls=":", alpha=0.7)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()

def plot_thd_figure(fs, T, f1, f2, vrms_list, params, preset_name):
    """Stellt den frequenzabhaengigen THD des Forward-Modells grafisch dar."""
    plt.figure(figsize=(10, 6))

    for vrms in vrms_list:
        freqs, thd_db, _, _ = compute_stationary_sine_thd_curve(
            fs,
            params,
            vrms,
            f1,
            f2,
            num_points=max(24, int(T * 6)),
        )
        mask = (freqs >= f1) & (freqs <= f2) & np.isfinite(thd_db)
        plt.semilogx(freqs[mask], thd_db[mask], label=f"{vrms:.1f} Vrms")

    plt.title(f"Frequenzabhaengiger THD des Bernardini-Forward-Modells ({preset_name})")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("THD [dB]")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()

def plot_psd_figure(fs, f1, f2, preset_name, sweep_results):
    """Stellt die Sweep-PSD fuer einen oder mehrere Pegel grafisch dar."""
    plt.figure(figsize=(10, 6))

    for vrms, result in sweep_results.items():
        output = prepare_analysis_signal(result["output"], fs)
        freqs, psd_db = compute_psd_db(output, fs)
        mask = (freqs >= f1) & (freqs <= f2)
        plt.semilogx(freqs[mask], psd_db[mask], label=f"{vrms:.1f} Vrms")

    plt.title(f"PSD(Pout) des Bernardini-Forward-Modells ({preset_name})")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD(Pout) [dB/Hz]")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()

def plot_power_spectrum_figure(fs, vrms_list, params, preset_name):
    """Stellt das Leistungsspektrum eines Bernardini-Sinusfalls grafisch dar."""
    duration_s = 1.0
    settle_s = 0.2

    plt.figure(figsize=(10, 6))

    for vrms in vrms_list:
        tone, analysis_freq_hz, resonance_hz = make_verification_sine(fs, params, vrms, duration_s)
        model = BernardiniSpeakerSolver(fs, params)
        output = simulate_model(model, tone)

        analysis_signal = prepare_analysis_signal(output, fs, settle_s=settle_s, tone_freqs_hz=[analysis_freq_hz])
        freqs, power_db = compute_power_spectrum_db(analysis_signal, fs)
        mask = (freqs >= 10.0) & (freqs <= min(5000.0, fs / 2.0))
        plt.semilogx(freqs[mask], power_db[mask], label=f"{vrms:.1f} Vrms")

    plt.title(
        f"Leistungsspektrum von Pout ({preset_name}, fin ≈ fr/2 = {analysis_freq_hz:.1f} Hz)"
    )
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Power(Pout) [dB]")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()

def plot_single_tone_verification(
    fs,
    params,
    vrms,
    preset_name,
    duration_s=1.0,
    settle_s=0.2,
    freq_hz=None,
    tone=None,
    output=None,
):
    """Erzeugt die kombinierte Fig.-6-aehnliche Einzeltone-Verifikation."""
    resonance_hz = estimate_small_signal_resonance(params)

    if tone is None:
        if freq_hz is None:
            tone, fin_hz, resonance_hz = make_verification_sine(fs, params, vrms, duration_s)
        else:
            fin_hz = float(freq_hz)
            tone = generate_sine(fs, duration_s, fin_hz, vrms)
    else:
        fin_hz = float(freq_hz) if freq_hz is not None else 0.0

    if output is None:
        model = BernardiniSpeakerSolver(fs, params)
        output = simulate_model(model, tone)

    t = np.arange(len(tone)) / fs
    analysis_signal = prepare_analysis_signal(output, fs, settle_s=settle_s, tone_freqs_hz=[fin_hz])
    freqs, power_db = compute_power_spectrum_db(analysis_signal, fs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(t, tone, label="Eingang Vin(t)")
    axes[0].plot(t, output, label="Ausgang Pout(t)", alpha=0.8)
    axes[0].set_title(
        f"Sinus-Anregung fuer Modellverifikation ({preset_name}, fin = {fin_hz:.1f} Hz, fr = {resonance_hz:.1f} Hz)"
    )
    axes[0].set_xlabel("Zeit [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, ls=":")
    axes[0].legend()

    mask = (freqs >= 10.0) & (freqs <= min(5000.0, fs / 2.0))
    axes[1].semilogx(freqs[mask], power_db[mask], label=f"{vrms:.1f} Vrms")
    axes[1].set_title("Leistungsspektrum des Ausgangssignals nach Einschwingvorgang")
    axes[1].set_xlabel("Frequenz [Hz]")
    axes[1].set_ylabel("Power(Pout) [dB]")
    axes[1].grid(True, which="both", ls=":")
    axes[1].legend()

    fig.tight_layout()

def plot_signal_psd(signal, fs, title, fmin=10.0, fmax=None):
    """Zeigt die PSD eines vorbereiteten Signals an."""
    freqs, psd_db = compute_psd_db(signal, fs)
    upper = fs / 2.0 if fmax is None else min(fmax, fs / 2.0)
    mask = (freqs >= fmin) & (freqs <= upper)

    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs[mask], psd_db[mask])
    plt.title(title)
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD(Pout) [dB/Hz]")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()

def plot_signal_highres_psd(signal, fs, title, fmin=10.0, fmax=None):
    """Zeigt eine hochaufgeloeste PSD eines vorbereiteten Signals an."""
    freqs, psd_db = compute_highres_psd_db(signal, fs)
    upper = fs / 2.0 if fmax is None else min(fmax, fs / 2.0)
    mask = (freqs >= fmin) & (freqs <= upper)

    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs[mask], psd_db[mask])
    plt.title(title)
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD(Pout) [dB/Hz]")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()

def plot_signal_power_spectrum(signal, fs, title, fmin=10.0, fmax=None):
    """Zeigt das Leistungsspektrum eines vorbereiteten Signals an."""
    freqs, power_db = compute_power_spectrum_db(signal, fs)
    upper = fs / 2.0 if fmax is None else min(fmax, fs / 2.0)
    mask = (freqs >= fmin) & (freqs <= upper)

    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs[mask], power_db[mask])
    plt.title(title)
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Power(Pout) [dB]")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()

def run_sweep_batch(fs, T, f1, f2, vrms_list, params):
    """Simuliert mehrere Sweep-Faelle fuer unterschiedliche Eingangspegel."""
    results = {}
    for vrms in vrms_list:
        sweep, inv = generate_sweep(fs, T, f1, f2, vrms)
        model = BernardiniSpeakerSolver(fs, params)
        output = simulate_model(model, sweep)
        results[vrms] = {
            "sweep": sweep,
            "inv": inv,
            "output": output,
        }
    return results

# Laden gespeicherter Anregungssignale

def load_metadata_json(metadata_path):
    """Laedt die Metadaten eines gespeicherten Inverslaufs."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))

def load_drive_signal_bundle(path_text):
    """Laedt ein gespeichertes Drive-Signal samt optionaler Metadaten."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if path.is_dir():
        drive_path = path / "inverse_drive_vin_float32.npy"
        metadata_path = path / "metadata.json"
        bundle_root = path
    else:
        drive_path = path
        metadata_path = path.parent / "metadata.json"
        bundle_root = path.parent

    if not drive_path.exists():
        raise FileNotFoundError(f"Drive-Datei nicht gefunden: {drive_path}")

    signal = np.asarray(np.load(drive_path), dtype=float)
    metadata = load_metadata_json(metadata_path)
    return {
        "bundle_root": bundle_root,
        "drive_path": drive_path,
        "metadata_path": metadata_path,
        "signal": signal,
        "metadata": metadata,
    }

def discover_inverse_run_directories(root_text):
    """Sucht Inverslauf-Unterordner in einem Wurzelverzeichnis."""
    root = Path(root_text).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    if root.is_file():
        return [root.parent]

    candidates = {match.parent for match in root.rglob("inverse_drive_vin_float32.npy")}
    if (root / "inverse_drive_vin_float32.npy").exists():
        candidates.add(root)
    return sorted(candidates)

def print_harmonic_table(mode, harmonic_data):
    """Gibt eine kleine Tabelle der gemessenen Harmonischen aus."""
    print(f"\nHARMONICS [{mode}]")
    print("k\tfreq_bin[Hz]\tpower[dB]\trel_to_H1[dB]")
    for item in harmonic_data:
        print(
            f"{item['harmonic']}\t{item['bin_hz']:.3f}\t\t{item['power_db']:.3f}\t\t{item['relative_db']:.3f}"
        )

def print_loaded_signal_summary(label, freq_hz, fs, preset_name, thd_point):
    """Gibt die Eckdaten eines geladenen Signals mit THD-Messung aus."""
    print("=" * 80)
    print(label)
    print(f"Preset         : {preset_name}")
    print(f"fs             : {fs} Hz")
    print(f"Fundamental    : {freq_hz:.6g} Hz")
    print(f"THD            : {thd_point['thd_db']:.3f} dB")
    dominant = thd_point["dominant_harmonic"]
    if dominant is not None:
        print(
            f"Dominant Harm. : H{dominant} ({thd_point['dominant_relative_db']:.3f} dB relativ zu H1)"
        )
    print("=" * 80)

def run_loaded_signal_thd_case(
    bundle,
    preset_name,
    fs,
    freq_hz,
    harmonics=5,
    settle_s=None,
    show_plot=True,
):
    """Simuliert ein geladenes Drive-Signal und misst dessen THD im Forward-Modell."""
    params = dict(BERNARDINI_PRESETS[preset_name])
    drive_signal = np.asarray(bundle["signal"], dtype=float)
    output = simulate_model(BernardiniSpeakerSolver(fs, params), drive_signal)

    if settle_s is None:
        _, settle_s = stationary_tone_timing(freq_hz)

    thd_point = compute_output_thd_point(
        output,
        fs,
        freq_hz,
        harmonics=harmonics,
        settle_s=settle_s,
    )

    print_loaded_signal_summary("THD AUS GELADENEM DRIVE-SIGNAL", freq_hz, fs, preset_name, thd_point)
    print_harmonic_table("loaded-file", thd_point["harmonic_data"])

    if show_plot:
        plot_signal_power_spectrum(
            thd_point["analysis_signal"],
            fs,
            (
                f"Power(Pout) fuer geladenes Drive-Signal "
                f"({preset_name}, fin = {freq_hz:.2f} Hz, THD = {thd_point['thd_db']:.2f} dB)"
            ),
        )

    return {
        "thd_point": thd_point,
        "output_signal": output,
    }

def run_loaded_signal_thd_curve_from_directory(root_text, harmonics=5, show_plot=True):
    """Erzeugt eine THD-Kurve aus mehreren gespeicherten Drive-Signalen."""
    run_dirs = discover_inverse_run_directories(root_text)
    if not run_dirs:
        raise FileNotFoundError(f"Keine Invers-Run-Ordner gefunden unter: {root_text}")

    results = []
    print("=" * 80)
    print("THD-KURVE AUS GELADENEN DRIVE-SIGNALEN")
    print(f"Runs gefunden   : {len(run_dirs)}")
    print("=" * 80)
    for run_dir in run_dirs:
        bundle = load_drive_signal_bundle(str(run_dir))
        metadata = bundle["metadata"]
        preset_name = metadata.get("preset_name", "Spk-1")
        fs = int(metadata.get("fs_hz", 96000))
        freq_hz = float(metadata["freq_hz"])
        settle_s = None
        if "duration_s" in metadata:
            _, settle_s = stationary_tone_timing(freq_hz)

        case_result = run_loaded_signal_thd_case(
            bundle,
            preset_name=preset_name,
            fs=fs,
            freq_hz=freq_hz,
            harmonics=harmonics,
            settle_s=settle_s,
            show_plot=False,
        )
        results.append(
            {
                "run_dir": run_dir,
                "preset_name": preset_name,
                "fs": fs,
                "freq_hz": freq_hz,
                "thd_db": case_result["thd_point"]["thd_db"],
            }
        )

    results.sort(key=lambda item: item["freq_hz"])
    freqs = np.asarray([item["freq_hz"] for item in results], dtype=float)
    thd_db = np.asarray([item["thd_db"] for item in results], dtype=float)

    print("\nTHD CURVE")
    print("freq[Hz]\tTHD[dB]\tpreset")
    for item in results:
        print(f"{item['freq_hz']:.6g}\t{item['thd_db']:.3f}\t{item['preset_name']}")

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.semilogx(freqs, thd_db, marker="o")
        plt.title("THD-Kurve aus geladenen Invers-Drive-Signalen")
        plt.xlabel("Frequenz [Hz]")
        plt.ylabel("THD [dB]")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()

    return results

# Interaktive Eingabe und Programmstart

def _prompt_text(prompt, default_text):
    """Fragt einen Textwert mit Defaultwert ab."""
    raw = input(f"{prompt} [{default_text}]: ").strip()
    return raw if raw else str(default_text)

def _prompt_float(prompt, default_value):
    """Fragt einen numerischen Wert mit Defaultwert ab."""
    raw = input(f"{prompt} [{default_value}]: ").strip()
    return float(raw) if raw else float(default_value)

def _prompt_signal_choice():
    """Fragt die gewuenschte Hauptauswertung im Startmenue ab."""
    print("Eingangssignal waehlen:")
    print("  1 = Sinus")
    print("  2 = Sweep")
    print("  3 = Two-Tone-Sinus")
    print("  4 = Gespeichertes Drive-Signal (.npy / Invers-Run-Ordner)")
    print("  5 = Nichtlinearitaeten ueber x (Fig.-5-artig)")
    choice = input("Auswahl [2]: ").strip().lower()

    if choice in ("", "2", "sweep"):
        return "sweep"
    if choice in ("1", "sinus", "sine"):
        return "sine"
    if choice in ("3", "two-tone", "twotone", "two tone"):
        return "two-tone"
    if choice in ("4", "file", "npy", "inverse"):
        return "file"
    if choice in ("5", "nonlinear", "nonlinearities", "fig5", "x"):
        return "nonlinearities"
    raise ValueError("Ungueltige Signalauswahl.")

def _prompt_plot_choice(signal_kind):
    """Fragt die zur Signalklasse passende Detailauswertung ab."""
    if signal_kind == "sweep":
        print("Plot waehlen:")
        print("  1 = x: Frequenz [Hz], y: THD [dB] (stationaere Sinusmessung)")
        print("  2 = x: Frequenz [Hz], y: PSD(Pout) [dB]")
        choice = input("Auswahl [1]: ").strip().lower()
        if choice in ("", "1", "thd"):
            return "thd"
        if choice in ("2", "psd"):
            return "psd"
    elif signal_kind == "sine":
        print("Plot waehlen:")
        print("  1 = Fig.-6-Verifikation: x: Zeit [s] / Frequenz [Hz], y: Amplitude / Power(Pout) [dB]")
        print("  2 = x: Frequenz [Hz], y: Power(Pout) [dB]")
        print("  3 = x: Frequenz [Hz], y: PSD(Pout) [dB/Hz]")
        print("  4 = Bernardini-Sinusplot: x: Frequenz [Hz], y: Power(Pout) [dB], fin ~= fr/2")
        choice = input("Auswahl [1]: ").strip().lower()
        if choice in ("", "1", "verification", "fig6"):
            return "verification"
        if choice in ("2", "power"):
            return "power"
        if choice in ("3", "psd"):
            return "psd"
        if choice in ("4", "bernardini", "paper", "fr2"):
            return "paper-power"
    elif signal_kind == "two-tone":
        print("Plot waehlen:")
        print("  1 = x: Frequenz [Hz], y: PSD(Pout) [dB/Hz] (hochaufloesend, Fig.-7-nah)")
        print("  2 = x: Frequenz [Hz], y: Power(Pout) [dB]")
        choice = input("Auswahl [1]: ").strip().lower()
        if choice in ("", "1", "psd"):
            return "psd"
        if choice in ("2", "power"):
            return "power"
    elif signal_kind == "file":
        print("Auswertung waehlen:")
        print("  1 = THD fuer einen einzelnen Invers-Run")
        print("  2 = THD-Kurve aus mehreren Invers-Runs in einem Ordner")
        choice = input("Auswahl [1]: ").strip().lower()
        if choice in ("", "1", "thd", "single"):
            return "file-thd-single"
        if choice in ("2", "curve", "folder"):
            return "file-thd-curve"
    elif signal_kind == "nonlinearities":
        return "curves"

    raise ValueError("Ungueltige Plot-Auswahl.")

def interactive_main():
    """Steuert den interaktiven Standardablauf des Programms."""
    signal_kind = _prompt_signal_choice()
    plot_kind = _prompt_plot_choice(signal_kind)

    if signal_kind == "sweep":
        active_preset = _prompt_text("Preset", "Spk-1")
        if active_preset not in BERNARDINI_PRESETS:
            raise ValueError(f"Unbekanntes Preset: {active_preset}")
        params = dict(BERNARDINI_PRESETS[active_preset])
        fs = int(_prompt_float("Samplingrate fs", 48000))
        vrms = _prompt_float("Eingangspegel Vrms", 11.5)
        T = _prompt_float("Sweep-Dauer T [s]", 4.0)
        f1 = _prompt_float("Startfrequenz f1 [Hz]", 20.0)
        f2 = _prompt_float("Endfrequenz f2 [Hz]", 20000.0)

        if plot_kind == "thd":
            plot_thd_figure(fs, T, f1, f2, [vrms], params, active_preset)
        else:
            sweep_results = run_sweep_batch(fs, T, f1, f2, [vrms], params)
            plot_psd_figure(fs, f1, f2, active_preset, sweep_results)

    elif signal_kind == "sine":
        active_preset = _prompt_text("Preset", "Spk-1")
        if active_preset not in BERNARDINI_PRESETS:
            raise ValueError(f"Unbekanntes Preset: {active_preset}")
        params = dict(BERNARDINI_PRESETS[active_preset])
        fs = int(_prompt_float("Samplingrate fs", 48000))
        vrms = _prompt_float("Eingangspegel Vrms", 11.5)
        duration_s = _prompt_float("Sinus-Dauer [s]", 1.0)
        freq_hz = _prompt_float("Sinusfrequenz [Hz]", 50.0)
        settle_s = _prompt_float("Einschwingzeit [s]", 0.2)

        if plot_kind == "paper-power":
            plot_power_spectrum_figure(fs, [vrms], params, active_preset)
        else:
            tone = generate_sine(fs, duration_s, freq_hz, vrms)
            model = BernardiniSpeakerSolver(fs, params)
            output = simulate_model(model, tone)
            analysis_signal = prepare_analysis_signal(output, fs, settle_s=settle_s, tone_freqs_hz=[freq_hz])

            if plot_kind == "verification":
                plot_single_tone_verification(
                    fs,
                    params,
                    vrms,
                    active_preset,
                    duration_s=duration_s,
                    settle_s=settle_s,
                    freq_hz=freq_hz,
                    tone=tone,
                    output=output,
                )
            elif plot_kind == "psd":
                plot_signal_psd(
                    analysis_signal,
                    fs,
                    f"PSD(Pout) fuer Sinus-Anregung ({active_preset}, fin = {freq_hz:.1f} Hz)",
                )
            else:
                plot_signal_power_spectrum(
                    analysis_signal,
                    fs,
                    f"Power(Pout) fuer Sinus-Anregung ({active_preset}, fin = {freq_hz:.1f} Hz)",
                )

    elif signal_kind == "two-tone":
        active_preset = _prompt_text("Preset", "Spk-1")
        if active_preset not in BERNARDINI_PRESETS:
            raise ValueError(f"Unbekanntes Preset: {active_preset}")
        params = dict(BERNARDINI_PRESETS[active_preset])
        paper_defaults = get_paper_two_tone_defaults(active_preset)
        fs = int(_prompt_float("Samplingrate fs", 48000))
        vrms = _prompt_float("Eingangspegel Vrms", paper_defaults["vrms"])
        duration_s = _prompt_float("Two-Tone-Dauer [s]", 1.0)
        freq1_hz = _prompt_float("Frequenz 1 [Hz]", paper_defaults["freq1_hz"])
        freq2_hz = _prompt_float("Frequenz 2 [Hz]", paper_defaults["freq2_hz"])
        phase1_rad = _prompt_float("Phase 1 [rad]", paper_defaults["phase1_rad"])
        phase2_rad = _prompt_float("Phase 2 [rad]", paper_defaults["phase2_rad"])
        settle_s = _prompt_float("Einschwingzeit [s]", 0.2)

        signal = generate_two_tone(
            fs,
            duration_s,
            freq1_hz,
            freq2_hz,
            vrms,
            phase1_rad=phase1_rad,
            phase2_rad=phase2_rad,
        )
        model = BernardiniSpeakerSolver(fs, params)

        output = simulate_model(model, signal)
        analysis_signal = prepare_two_tone_analysis_signal(output, fs, settle_s=settle_s)
        title_suffix = (
            f"({active_preset}, f1 = {freq1_hz:.2f} Hz, f2 = {freq2_hz:.2f} Hz, "
            f"phi1 = {phase1_rad:.4f} rad, phi2 = {phase2_rad:.4f} rad)"
        )

        if plot_kind == "psd":
            plot_signal_highres_psd(
                analysis_signal,
                fs,
                f"PSD(Pout) fuer Two-Tone-Anregung {title_suffix}",
            )
        else:
            plot_signal_power_spectrum(
                analysis_signal,
                fs,
                f"Power(Pout) fuer Two-Tone-Anregung {title_suffix}",
            )
    elif signal_kind == "nonlinearities":
        active_preset = _prompt_text("Preset", "Spk-1")
        if active_preset not in BERNARDINI_PRESETS:
            raise ValueError(f"Unbekanntes Preset: {active_preset}")
        params = dict(BERNARDINI_PRESETS[active_preset])
        x_default_min_mm, x_default_max_mm = get_fig5_displacement_defaults_mm(active_preset)
        x_min_mm = _prompt_float("x-Minimum [mm]", x_default_min_mm)
        x_max_mm = _prompt_float("x-Maximum [mm]", x_default_max_mm)
        num_points = int(_prompt_float("Anzahl Stuetzstellen", 801))
        plot_nonlinearities_vs_displacement(
            params,
            active_preset,
            x_min_mm,
            x_max_mm,
            num_points=max(num_points, 32),
        )
    else:
        harmonics = int(_prompt_float("Anzahl Harmonische fuer THD", 5))
        if plot_kind == "file-thd-single":
            bundle_path = _prompt_text("Pfad zur .npy-Datei oder zum Invers-Run-Ordner", "inverse_outputs")
            bundle = load_drive_signal_bundle(bundle_path)
            metadata = bundle["metadata"]
            if "preset_name" in metadata:
                preset_name = str(metadata["preset_name"])
            else:
                preset_name = _prompt_text("Preset fuer das geladene Signal", "Spk-1")
            if preset_name not in BERNARDINI_PRESETS:
                raise ValueError(f"Unbekanntes Preset: {preset_name}")

            if "fs_hz" in metadata:
                fs_loaded = int(metadata["fs_hz"])
            else:
                fs_loaded = int(_prompt_float("Samplingrate fs des geladenen Signals", 96000))
            if "freq_hz" in metadata:
                freq_hz = float(metadata["freq_hz"])
            else:
                freq_hz = _prompt_float("Grundfrequenz des geladenen Signals [Hz]", 50.0)
            settle_default = stationary_tone_timing(freq_hz)[1]
            settle_s = _prompt_float("Einschwingzeit fuer THD-Auswertung [s]", settle_default)
            run_loaded_signal_thd_case(
                bundle,
                preset_name=preset_name,
                fs=fs_loaded,
                freq_hz=freq_hz,
                harmonics=harmonics,
                settle_s=settle_s,
                show_plot=True,
            )
        else:
            root_path = _prompt_text("Pfad zum Ordner mit mehreren Invers-Runs", "inverse_outputs")
            run_loaded_signal_thd_curve_from_directory(
                root_path,
                harmonics=harmonics,
                show_plot=True,
            )

    plt.show()

if __name__ == "__main__":
    interactive_main()
