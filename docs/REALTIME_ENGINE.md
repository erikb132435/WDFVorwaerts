# Virtueller Echtzeittest des Fast Solvers

`src/RealtimeWDFEngine.py` ist die GUI-unabhaengige erste Echtzeitstufe. Eine
WAV-Datei wird in festen Bloecken gelesen und mit dem zustandsbehafteten
`FastBernardiniSpeakerSolver` verarbeitet. Die optionale Zeitfreigabe bildet
die Blockperioden der WAV-Abtastrate nach; sie ersetzt noch keinen echten
Audiotreiber-Callback.

## GUI-Reiter

`src/WDFSimulatorGUI.py` stellt dieselbe Engine im Reiter
`Virtueller Echtzeittest` bereit. Dort koennen WAV-Datei, Lautsprecherpreset,
Blockgroesse, Nennpegel, Pegelkonvention, optionale Laufzeitbegrenzung,
Echtzeittaktung, Garbage-Collector-Verhalten, Ausgabepuffer, Monitor-WAV und
Ausgabeordner eingestellt werden. Fuer Diagnoselaeufe ist der RAM-Puffer in
der GUI voreingestellt; die wissenschaftliche NPY-Datei wird dabei erst nach
der zeitkritischen Blockschleife geschrieben.

Der Lauf arbeitet in einem Hintergrundthread. Die GUI erhaelt hoechstens etwa
zehn Fortschrittsmeldungen pro Sekunde und fuehrt weder WDF-Berechnung noch
Dateischreiben im Tkinter-Thread aus. `Abbrechen` setzt ein Thread-Ereignis; die
Engine beendet den Lauf kontrolliert zwischen zwei Bloecken und entfernt eine
unvollstaendige Druckdatei. Nach erfolgreichem Abschluss zeigt die GUI Kernel-,
Gesamtblock-, Jitter- und relative Threadarbeitsverlaeufe sowie alle drei
Deadline-Zaehler an.

## Beispiel

```powershell
& "D:\PsychoPy\python.exe" src\RealtimeWDFEngine.py input.wav `
  --preset Spk-2 `
  --block-size 128 `
  --target-rms-voltage 3.0 `
  --output-storage memory `
  --output-root data\RealtimeTests
```

`--target-rms-voltage` ist die tatsaechliche RMS-Spannung, die der Solver
erhaelt. Eine Nennpegelkonvention wie `A_in * sqrt(2)` muss vor diesem Aufruf
explizit auf die effektive Spannung abgebildet werden. Mit
`--preserve-wav-level` wird nicht auf einen RMS-Wert normiert; dann bestimmt
`--volts-per-full-scale` die Spannungsabbildung.

Der virtuelle Taktgeber wartet im letzten Bereich vor einer Blockfreigabe aktiv,
weil Windows sehr kurze `sleep()`-Aufrufe teilweise um mehrere Millisekunden
ueberschreitet. Diese aktive Wartezeit gehoert nicht zur gemessenen Blockzeit.
Ein spaeterer echter Audio-Callback wird dagegen vom Audiotreiber freigegeben.

Weitere wichtige Optionen:

- `--unpaced`: maximal schnelle Blockverarbeitung ohne reale Zeitfreigabe.
- `--maximum-duration 10`: nur die ersten zehn Sekunden untersuchen.
- `--disable-gc`: Python-Garbage-Collector waehrend der Blockschleife
  voruebergehend deaktivieren.
- `--output-storage memory`: Druckausgang waehrend des Streams im RAM halten
  und erst danach speichern. `memmap` behaelt die bisherige direkte
  dateibasierte Ausgabe bei. Der Float64-Puffer benoetigt acht Byte pro Sample,
  also bei 44,1 kHz etwa 21,2 MB pro Minute und Kanal.
- `--no-monitor-wav`: keine normalisierte Hoerdatei erzeugen.

## Ergebnisdateien

- `output_pressure_pa.npy`: wissenschaftlicher Float64-Ausgang in Pascal. Im
  Modus `memmap` wird die Datei waehrend der Blockverarbeitung gefuellt; im
  Modus `memory` wird sie erst nach Ende der Zeitmessung geschrieben.
- `block_timing.csv`: ein Datensatz pro Block mit Eingangsvorbereitung,
  Kernelzeit, gesamter Blockzeit, dem zugehoerigen Threadarbeitszaehler,
  Startlatenz und drei Deadline-Kriterien.
- `run_metadata.json`: Eingangshash, Pegelabbildung, Modellparameter,
  Solverversion, Konfiguration und zusammengefasste Laufzeitstatistik.
- `monitor_normalized.wav`: separat auf PCM16-Vollaussteuerung normalisierte
  Hoerdatei ohne wissenschaftlich gueltigen Absolutpegel.

## Deadline-Kriterien

- `kernel_deadline_miss`: nur `process_block_into()` war langsamer als die
  Blockfrist.
- `total_processing_deadline_miss`: WAV-Konvertierung, Spannungsskalierung und
  Solver waren gemeinsam langsamer als die Blockfrist.
- `scheduled_deadline_miss`: bei aktivem Pacing wurde die absolute Deadline
  unter Einbeziehung des Startjitters ueberschritten.

Ein Lauf gilt noch nicht allein deshalb als echtzeitfaehig, weil die mittlere
Zeit unter der Deadline liegt. Fuer einen belastbaren Nachweis sind lange
Laeufe ohne Deadline-Verletzung sowie P99-/Maximalwerte unter zusaetzlicher
System- und GUI-Last notwendig.

Die zusammengefasste Statistik enthaelt zusaetzlich p99,9 und gruppiert
aufeinanderfolgende Deadline-Verletzungen in Episoden. Dadurch wird eine
einzelne Stoerung mit mehreren Folgefehlern nicht als mehrere unabhaengige
Ursachen interpretiert.

## Threadarbeitsdiagnose

Die bisherige Wandzeit laeuft auch dann weiter, wenn Windows den Worker-Thread
zwischenzeitlich nicht ausfuehrt. `time.thread_time()` erwies sich unter der
verwendeten Windows-Umgebung mit effektiven 15,625-ms-Schritten als zu grob
fuer Audiobloecke. Deshalb wird unter Windows `QueryThreadCycleTime` als
hochaufloesender Zaehler der ausgefuehrten CPU-Zyklen verwendet. Auf anderen
Plattformen dient `time.thread_time_ns()` als Arbeitszaehler.

- Hohe relative Wandzeit bei normaler relativer Threadarbeit weist auf Warten oder
  Descheduling hin.
- Steigen Wandzeit und Threadarbeit gemeinsam, liegt die Ursache eher im
  tatsaechlichen Rechenpfad.

Wegen dynamischer CPU-Frequenzen werden Zyklen nicht in eine scheinbar exakte
Zeit umgerechnet. GUI und Metadaten geben P99 und Maximum deshalb relativ zum
jeweiligen Median an. Auch dieser Vergleich ist ein Diagnoseindikator und keine
eindeutige Ursachenklassifikation.

## Aktuelle Grenze

Die Engine testet WAV-Streaming, Zustandsuebergabe, Puffernutzung und
Zeitverhalten und ist in die GUI eingebunden. Sie besitzt noch keinen echten
Audiointerface-Callback; dieser folgt erst auf Basis des hier getesteten
Blockpfads.
