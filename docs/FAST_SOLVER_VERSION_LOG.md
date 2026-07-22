# Versionslog: paralleler schneller WDF-Solver

Dieses Dokument beschreibt ausschliesslich den optimierten Modellpfad in
`src/WDFSolverFast.py`. Die wissenschaftliche Referenzimplementierung
`src/WDFSolverVPres.py` bleibt davon getrennt und wird nicht durch den schnellen
Solver ersetzt.

## Version 0.2.0 – 22.07.2026

### Parallele GUI-Einbindung

- Im bestehenden Simulationsreiter eine Solverauswahl mit den Modi
  `Referenzsolver`, `Schneller Python-Solver` und `Beide Solver vergleichen`
  ergaenzt. Untersuchungs-, Signal-, Pegel- und Referenzparameter bleiben fuer
  alle Modi identisch.
- Im Vergleichsmodus wird das Eingangssignal einmal erzeugt und unveraendert an
  zwei frisch initialisierte Solverinstanzen uebergeben. Die schnelle Ausgabe
  dient als Primaerausgabe fuer bestehende Messdaten- und Spektralvergleiche.
- Referenzausgabe, schnelle Ausgabe, Differenzsignal, Rechenzeiten,
  Beschleunigungsfaktor, maximaler Samplefehler, RMSE und relativer L2-Fehler im
  Simulationsergebnis und beim Datenexport mitgefuehrt.
- Darstellung `Solververgleich` mit ueberlagerten Ausgaengen und separatem
  Differenzdiagramm ergaenzt. Zeitverlaeufe, Leistungs- und
  Leistungsdichtespektren sowie optional aufgezeichnete interne Zustaende koennen
  beide Solver gemeinsam darstellen.
- Den THD-Frequenzgang auf den gemeinsamen Solveradapter umgestellt. Im
  Vergleichsmodus werden fuer jedes einmalig erzeugte Sinussignal beide
  Solverausgaenge und zwei THD-Kurven berechnet.
- Solverzeiten getrennt von Signalerzeugung, Spektralanalyse, Messdatenvergleich
  und Plotaufbau ausgewiesen.

### Verifikation der GUI-Integration

- Zwoelf automatisierte Tests erfolgreich: sieben Kernelregressionstests und
  fuenf Tests fuer Backendadapter, Ergebnisstruktur, THD-Reihe und
  GUI-unabhaengiges Rendern des Solververgleichs.
- Vollstaendiger Spk-2-Zweitonlauf mit 96 kHz, einer Sekunde Signaldauer,
  internen Spuren und empirischem Reproduktionspegel:

  - Referenzsolver: 12,92 s,
  - schneller Solver: 0,554 s,
  - Beschleunigung: Faktor 23,32,
  - maximale Druckabweichung: $2{,}78\cdot10^{-15}\,\mathrm{Pa}$,
  - relativer L2-Fehler: $1{,}72\cdot10^{-15}$.

Die GUI-Einbindung ist weiterhin ein Offline-Simulationsvergleich. Sie fuehrt
noch keinen Audiocallback aus und stellt deshalb keinen zusaetzlichen
Echtzeitnachweis unter Audio- oder GUI-Last dar.

## Version 0.1.0 – 22.07.2026

### Modell- und Schnittstellenstruktur

- Eigenstaendigen `FastBernardiniSpeakerSolver` mit derselben Parameterstruktur
  `solver(fs, params)` wie beim Referenzsolver eingefuehrt.
- Zustandsbehaftete Blockverarbeitung implementiert. Der Endzustand eines Blocks
  ist der Anfangszustand des Folgeblocks; die Blockaufteilung veraendert das
  Ausgangssignal nicht.
- Kompatiblen Einsprung `simulate_model(model, input_signal, record_traces=False)`
  sowie eine Einzelsample-Schnittstelle `step(vin)` bereitgestellt.
- Optionalen Analysemodus mit den gleichen Ausgabegroessen wie im Referenzpfad
  umgesetzt: Eingangsspannung, Ausgangsdruck, Membranauslenkung,
  Membrangeschwindigkeit, Volumenstrom und Schwingspulenstrom.
- `process_block_into(input, output)` fuer die Wiederverwendung eines
  vorallokierten Float64-Ausgangspuffers ergaenzt.

### Rechnerische Optimierung

- Alle zeitkritischen WDF-Streurelationen skalar ausgewertet; innerhalb der
  Sampleschleife werden keine kleinen Streumatrizen oder Wellenvektoren erzeugt.
- Die Polynomkennlinien $Bl(x)$, $L_e(x)$ und $K_\mathrm{ms}(x)$ sowie deren
  Ableitungen mit festem Horner-Schema berechnet.
- Konstante Portwiderstaende und Polynomkoeffizienten bei der Initialisierung
  vorbereitet und innerhalb eines Blocks lokal gebunden.
- Parameterpruefungen aus unveraenderlichen Teilen der Sampleschleife entfernt;
  zustandsabhaengige Positivitaetsbedingungen fuer $L_e$, $K_\mathrm{ms}$ und
  deren Portwiderstaende bleiben aktiv.
- Rechenreihenfolge, Float64-Praezision, Backward-Euler-Zustaende,
  Vergessensfaktoren und nichtreziproke Root-Topologie gegenueber dem
  Referenzmodell nicht mathematisch approximiert.

### Numerische Verifikation

- Sieben automatisierte Regressionstests fuer beide Lautsprecher-Presets,
  hochpegelige Mehrtonsignale, interne Modellspuren, Resetverhalten,
  Einzelsample-Verarbeitung und vorallokierte Ausgangspuffer eingefuehrt.
- Blockgroessen von 1, 7, 32, 64, 127, 512 und 2048 Samples liefern exakt
  dieselbe Float64-Ausgabe wie die Verarbeitung als einzelner Gesamtblock.
- Bei einem einsekundigen Spk-2-Zweitonsignal mit 96 kHz wurden gegenueber dem
  Referenzsolver gemessen:

  $$
  \max_n\left|p_\mathrm{fast}[n]-p_\mathrm{ref}[n]\right|
  =2{,}78\cdot10^{-15}\,\mathrm{Pa},
  $$

  $$
  \frac{\lVert p_\mathrm{fast}-p_\mathrm{ref}\rVert_2}
  {\lVert p_\mathrm{ref}\rVert_2}
  =1{,}72\cdot10^{-15}.
  $$

### Laufzeitmessung

- Messumgebung: Python 3.8.10, NumPy 1.21.4, Windows, 24 000 Samples,
  Spk-2-Zweiton, 96 kHz, drei Wiederholungen.
- Referenzsolver: Median $137{,}60\,\mu\mathrm{s/Sample}$.
- Schneller Solver mit 64-Sample-Bloecken und vorallokiertem Ausgang:
  Median $5{,}73\,\mu\mathrm{s/Sample}$.
- Gemessene Beschleunigung: Faktor 24,0.
- Rechnerische Abtastfrist bei 96 kHz: $10{,}42\,\mu\mathrm{s/Sample}$;
  gemessene mittlere Reserve des Kernels: Faktor 1,82.
- Bei 375 einzeln gemessenen 64-Sample-Bloecken betrugen Median, P95, P99 und
  Maximum 386,4, 478,1, 554,2 und 559,7 Mikrosekunden. Die Blockfrist von
  666,7 Mikrosekunden wurde in diesem Lauf nicht ueberschritten.

Der Benchmark prueft den Python-Rechenkern unter Offline-Bedingungen. Er ist
noch kein Nachweis fuer einen stoerungsfreien Audiobetrieb unter GUI-Last oder
fuer harte Echtzeitanforderungen des Betriebssystems.

### Bekannte Grenzen und naechste Stufe

- Der schnelle Pfad akzeptiert derzeit Polynomkennlinien bis zum Grad vier.
  Freie Python-Kennlinienfunktionen verbleiben im Referenzsolver.
- `step()` ist aus Kompatibilitaetsgruenden vorhanden; der schnelle Pfad ist die
  blockweise Verarbeitung mit `process_block` beziehungsweise
  `process_block_into`.
- Numba ist in der gegenwaertigen Projektumgebung nicht installiert. Der Kernel
  bleibt deshalb in Version 0.1.0 rein in Python.
- Vor einer Audioanbindung folgen Dauerlauf-, Jitter- und Callback-Tests unter
  realer Systemlast. Eine spaetere Kompilierung des unveraenderten Blockkerns
  bleibt als zusaetzliche Sicherheitsreserve moeglich.
