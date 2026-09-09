# WDF-basierter Vorwärtssimulator für elektrodynamische Lautsprecher

Dieses Projekt implementiert das nichtlineare Lautsprecher-Ersatzschaltbild
nach Bernardini et al. als Wave-Digital-Filter-Modell (WDF). Es enthält eine
nachvollziehbare Referenzimplementierung, einen dazu numerisch geprüften
schnellen Solver sowie Werkzeuge für Simulation, Messdatenvergleich und
Laufzeituntersuchungen.

## Projektbestandteile

| Datei oder Ordner | Funktion |
|---|---|
| `src/WDFSolverVPres.py` | Wissenschaftlicher Referenzsolver und Lautsprecherparameter |
| `src/WDFSolverFast.py` | Ein-/ausgabegleicher, blockfähiger Fast Solver |
| `src/WDFSimulatorGUI.py` | Grafische Konfiguration, Auswertung und Ergebnisexport |
| `src/spectral_comparison.py` | Zeitliche Ausrichtung sowie Zeit- und Spektralvergleich mit Messdaten |
| `src/validation_export.py` | Reproduzierbarer Export eines physikalischen Validierungsfalls |
| `src/RealtimeWDFEngine.py` | Virtueller, blockweise getakteter Echtzeittest ohne Audiotreiber |
| `src/RealtimeAudioEngine.py` | GUI-unabhängiger Audiocallback-Kern |
| `src/PortAudioBackend.py` | Optionaler Hardwaretest über PortAudio beziehungsweise `sounddevice` |
| `tests/` | Regressions-, Export- und Echtzeittests |
| `benchmarks/` | Laufzeitvergleich von Referenz- und Fast Solver |
| `data/Bernardini_Messungen/` | Verwendete Messreferenzen, soweit im Projekt enthalten |
| `docs/` | Methodische Dokumentation und Versionsprotokolle |

## Installation

Benötigt werden Python, NumPy, SciPy und Matplotlib. Tkinter muss für die GUI
in der verwendeten Python-Installation verfügbar sein.

```powershell
python -m pip install numpy scipy matplotlib
```

Für den optionalen Test mit einem realen Audiogerät wird zusätzlich installiert:

```powershell
python -m pip install -r requirements-audio.txt
```

Alle nachfolgenden Befehle werden im Projektwurzelordner ausgeführt. Falls eine
bestimmte Python-Umgebung verwendet wird, kann `python` beispielsweise durch
`& "D:\PsychoPy\python.exe"` ersetzt werden.

## Grafische Simulation

Start:

```powershell
python src/WDFSimulatorGUI.py
```

Der Reiter **Wissenschaftliche Simulation** stellt insbesondere folgende
Untersuchungen bereit:

- Kleinsignal-Frequenzgänge aus Referenzdaten,
- auslenkungsabhängige Modellparameter,
- stationäre Sinusanregung und THD-Frequenzgang,
- Zweiton-Anregung,
- exponentieller Sinus-Sweep,
- externe WAV- oder NPY-Signale.

Die wesentlichen Eingaben sind:

- **Lautsprechermodell:** `Spk-1` oder `Spk-2`;
- **Solver:** Referenzsolver, Fast Solver oder direkter Vergleich beider;
- **Abtastrate:** zeitliche Diskretisierung des Modells;
- **Eingangspegel** $A_{\mathrm{in}}$: Parameter der elektrischen
  Eingangsspannung;
- **Eingangspegel-Konvention:** Interpretation und Skalierung dieses Parameters;
- **Signaldauer und Einschwingzeit:** erzeugter beziehungsweise ausgewerteter
  Zeitbereich;
- **versuchsspezifische Parameter:** beispielsweise Frequenz, Zweitonphasen
  oder Sweep-Grenzen;
- **Referenzdatei:** optionale Bernardini-Messung für den Vergleich.

### Eingangspegel

Im Modus **Standard** wird das Signal entsprechend

$$
V_{\mathrm{in}}(t)=A_{\mathrm{in}}
\frac{s_{\mathrm{in}}(t)}{\operatorname{rms}\!\left\{s_{\mathrm{in}}(t)\right\}}
$$

auf $\operatorname{rms}\!\left\{V_{\mathrm{in}}\right\}=A_{\mathrm{in}}$ normiert. Die
zusätzliche Option **Bernardini-Datensatz-Hypothese (zusätzlich × √2)** setzt
dagegen

$$
\operatorname{rms}\!\left\{V_{\mathrm{in}}\right\}=\sqrt{2}\,A_{\mathrm{in}}.
$$

Diese zweite Einstellung dient einer transparenten Sensitivitätsanalyse der
vorliegenden Messdaten. Sie ist keine aus dem Paper bestätigte Änderung des
Modells. Beide Solver erhalten unabhängig von der gewählten Signalart nur das
fertige Spannungssignal als Float64-Array.

### Ergebnisse und Export

Je nach Untersuchung können unter anderem Ausgangsdruck, Auslenkung,
Membrangeschwindigkeit, Schwingspulenstrom, PSD, Leistungsspektrum, Harmonische
und THD dargestellt werden.

- **Daten speichern:** schreibt numerische Arrays als `.npz` und skalare
  Metadaten als gleichnamige `.json`-Datei.
- **Plot speichern:** exportiert die aktuelle Darstellung als PNG, PDF oder SVG.
- **Validierungspaket:** steht bei einem Messdatenvergleich mit dem Fast Solver
  zur Verfügung und speichert Referenzkopien, Rohdaten, Vergleichstabellen,
  Kennwerte, Metadaten und den Plot in einem gemeinsamen Ordner.

Ein Validierungspaket übernimmt den bereits berechneten GUI-Vergleich und führt
keine nachträgliche Anpassung oder erneute Simulation durch. Einzelheiten stehen
in `docs/VALIDATION_EXPORT.md`.

## Direkte Verwendung des Fast Solvers

Der Solver unterscheidet selbst nicht zwischen Sinus, Zweiton oder Musik. Sein
Eingang ist stets ein eindimensionales Spannungssignal in Volt; sein Ausgang
ist der simulierte Druck in Pascal.

```python
import sys
import numpy as np

sys.path.insert(0, "src")

import WDFSolverFast as fast
import WDFSolverVPres as reference

fs_hz = 48_000.0
parameters = reference.BERNARDINI_PRESETS["Spk-2"]
time_s = np.arange(int(0.25 * fs_hz)) / fs_hz
input_voltage_v = np.sqrt(2.0) * 1.0 * np.sin(2.0 * np.pi * 81.4 * time_s)

solver = fast.FastBernardiniSpeakerSolver(fs_hz, parameters)
output_pressure_pa = solver.process_block(input_voltage_v)
```

Ein Solverobjekt ist zustandsbehaftet. Aufeinanderfolgende Blöcke müssen daher
mit derselben Instanz verarbeitet werden. `reset()` beginnt einen neuen Lauf
mit zurückgesetztem Modellzustand.

## Virtueller Echtzeittest

Der entsprechende GUI-Reiter oder folgender Terminalaufruf verarbeitet eine
WAV-Datei blockweise mit dem Fast Solver:

```powershell
python src/RealtimeWDFEngine.py "input.wav" `
  --preset Spk-2 `
  --block-size 256 `
  --target-rms-voltage 3.0 `
  --output-storage memory `
  --output-root data/RealtimeTests
```

Die WAV-Abtastrate wird aus der Datei übernommen. `--target-rms-voltage` ist
der tatsächliche RMS-Wert am Solvereingang. Die Engine speichert unter anderem
den Druckausgang, Blockzeiten, Deadline-Verletzungen und Laufmetadaten. Dieser
Test untersucht die blockweise Verarbeitung, verwendet aber noch keinen realen
Audiotreiber.

## Audiocallback-Test mit Hardware

Verfügbare Ausgabegeräte anzeigen:

```powershell
python src/PortAudioBackend.py --list-devices
```

Stummer und damit für erste Messungen empfohlener Lauf:

```powershell
python src/PortAudioBackend.py "input_48k.wav" `
  --device 10 `
  --sample-rate 48000 `
  --block-size 256 `
  --preset Spk-2 `
  --target-rms-voltage 3.0 `
  --maximum-duration 60
```

Ohne `--dry-output` oder `--pressure-output` wird Stille an das Audiogerät
ausgegeben, während der Modelldruck vollständig berechnet und nach Streamende
gespeichert wird. Hörbare Modi sind ausschließlich technische Monitorpfade und
keine kalibrierte Wiedergabe des simulierten Drucks. Weitere Sicherheitshinweise
und Optionen enthält `docs/REALTIME_AUDIO.md`.

## Tests und Laufzeitvergleich

Vollständige Testsuite:

```powershell
python -m unittest discover -s tests -v
```

Die Tests prüfen insbesondere die Übereinstimmung von Fast und Referenzsolver,
die Zustandskontinuität über Blockgrenzen, Exporte sowie virtuelle und reale
Callbackpfade.

Reproduzierbarer Solverbenchmark:

```powershell
python benchmarks/benchmark_wdf_solvers.py --samples 24000 --repeats 3 --block-size 256
```

## Wissenschaftliche Einordnung

Der Referenzsolver bleibt die maßgebliche Implementierung der übertragenen
WDF-Topologie. Der Fast Solver bildet denselben Rechenpfad effizienter ab und
wird durch Regressionstests gegen ihn geprüft. Die Übereinstimmung beider
Implementierungen ist eine numerische Verifikation; der Vergleich mit realen
Lautsprechermessungen ist davon getrennt als physikalische Validierung zu
behandeln.

Modellparameter, Eingangssignal, Abtastrate, Auswertebereich, zeitliche
Ausrichtung und verwendete Pegelskalierung müssen bei wissenschaftlichen Läufen
gemeinsam dokumentiert werden. Bereits erzeugte Ergebnisarchive werden durch
spätere Codeänderungen nicht automatisch aktualisiert.
