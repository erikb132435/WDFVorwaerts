# Audiocallback-Kern und PortAudio-Hardwaretest

`src/RealtimeAudioEngine.py` enthaelt den von GUI und Audiotreiber unabhaengigen
Callback-Kern. `src/PortAudioBackend.py` bindet ihn ueber `sounddevice` an einen
realen PortAudio-Ausgang an. Solver und Messpuffer bleiben dadurch identisch;
nur Aufrufzeitpunkt, Ausgabepuffer und Treiberstatus stammen von der Hardware.

## Aktueller Umfang

- 44,1 und 48 kHz,
- zustandsbehaftete Verarbeitung mit dem Fast Solver,
- feste Blockgroesse,
- vollstaendig vor dem Stream geladene und auf Spannung skalierte WAV,
- stummer, abgesenkter trockener oder geschuetzter Druckmonitor-Ausgang,
- vorallokierter Float64-Druckausgang,
- Callback-, Solver-, Threadarbeits- und Treiberstatusmetriken,
- Deadline-Episoden und p99,9,
- hardwarefreier simulierter Audiotreiber fuer Regressionstests,
- realer, headless ausgefuehrter PortAudio-Ausgangstest,
- getrennte Zeitmessung fuer Solverkern, Engine-Callback und gesamten
  PortAudio-Adaptercallback.

WAV-Samplerate und konfigurierte Samplerate muessen uebereinstimmen. Ein
Resampling waehrend des Callbacks ist ausdruecklich nicht vorgesehen.

## Verwendung ohne Audiohardware

```python
from RealtimeAudioEngine import AudioRealtimeConfig, run_simulated_wav

config = AudioRealtimeConfig(
    fs_hz=48000,
    block_size=256,
    preset_name="Spk-2",
    target_rms_voltage=3.0,
    output_mode="silent",
)

result = run_simulated_wav("input_48k.wav", config)
print(result.statistics.summary())
pressure_pa = result.output_pressure_pa
```

`output_mode="silent"` ist der sichere Standard. `output_mode="dry"` erzeugt
im simulierten Treiberausgang das unveraenderte, mit `dry_output_gain`
abgesenkte Eingangssignal. `output_mode="pressure"` gibt eine geschuetzte
Monitorabbildung des Modelldrucks aus; der wissenschaftliche Druck in Pascal
bleibt davon getrennt und unveraendert.

## Callback-Vertrag

Nach `prepare_wav()` kann ein spaeterer Audiotreiber fuer jeden Block
`process_callback(output_buffer, status)` aufrufen. Der Puffer muss genau
`block_size` Frames besitzen. Innerhalb dieses Aufrufs erfolgen keine
Dateizugriffe, GUI-Aufrufe oder Ergebnisarray-Allokationen.

Der letzte Teilblock wird im Audioausgang mit Nullen aufgefuellt. Nur die
tatsaechlich vorhandenen Eingangssamples werden an den Solver uebergeben;
dadurch entstehen keine kuenstlichen zusaetzlichen Modellzustaende.

## Simulierter Treiber

`SimulatedAudioBackend` ruft denselben Callback ohne reale Wartezeit auf. Er
prueft damit Datenfluss, Blockgrenzen, Zustandsuebergabe und Statusauswertung,
nicht jedoch Treiberscheduling oder reale Audio-Underruns. Kuenstliche
Treiberstatus koennen blockweise injiziert werden.

Die Regressionstests vergleichen den gesamten Callback-Druck samplegenau mit
dem Offline-Fast-Solver. Eine reale Hardwareanbindung darf diesen Vertrag nicht
veraendern.

## Reale Audiohardware

Die optionale Abhaengigkeit wird nur vom Hardwareadapter importiert. In der
PsychoPy-Projektumgebung kann die Verfuegbarkeit so geprueft werden:

```powershell
& "D:\PsychoPy\python.exe" -c "import sounddevice; print(sounddevice.__version__)"
```

Falls das Paket in einer anderen Python-Umgebung fehlt:

```powershell
python -m pip install -r requirements.txt
```

Zuerst werden die Ausgangsgeraete ohne Streamstart aufgelistet:

```powershell
& "D:\PsychoPy\python.exe" src\PortAudioBackend.py --list-devices
```

Danach kann ein stummer Hardwaretest gestartet werden. Die WAV-Samplerate muss
der mit `--sample-rate` gewaehlten Rate entsprechen:

```powershell
& "D:\PsychoPy\python.exe" src\PortAudioBackend.py input_48k.wav `
  --device 10 `
  --sample-rate 48000 `
  --block-size 256 `
  --preset Spk-2 `
  --target-rms-voltage 3.0 `
  --maximum-duration 60
```

Mit `--wasapi-exclusive` wird ein ausgewaehltes WASAPI-Geraet optional exklusiv
geoeffnet. Der bisherige Windows-Shared-Modus bleibt der Standard. Im
Exclusive-Modus darf das Geraet waehrend des Tests nicht von einer anderen
Anwendung belegt sein; ausserdem muss es die gewaehlte Samplerate direkt
unterstuetzen.

Ohne `--dry-output` oder `--pressure-output` schreibt der Callback
ausschliesslich Nullen in den Audioausgang. `--dry-output` gibt das abgesenkte
Originalsignal aus. In beiden hoerbaren Modi muss der physische Interfacepegel
vor Streambeginn auf Minimum stehen.

## Geschuetzter Druckmonitor

Der Druckmonitor wird explizit mit `--pressure-output` aktiviert. Die Abbildung
ist zeitinvariant und fuer jedes Sample definiert als

`digital[n] = clip(pressure_pa[n] / P_FS, -L, +L)`.

`P_FS` wird mit `--pressure-pa-per-full-scale` und `L` mit
`--output-peak-limit` eingestellt. Die sicheren Startwerte sind 20 Pa/FS und
0,1 beziehungsweise -20 dBFS maximale digitale Spitze. NaN wird zu Stille;
positive und negative Unendlichkeit werden auf die jeweilige Schutzgrenze
gesetzt. Es findet keine laufende Normalisierung, Pegelschaetzung oder
Vorausschau statt.

Beispiel fuer das Scarlett im Exclusive-Modus:

```powershell
& "D:\PsychoPy\python.exe" src\PortAudioBackend.py input_44k1.wav `
  --device 14 `
  --sample-rate 44100 `
  --block-size 256 `
  --preset Spk-2 `
  --target-rms-voltage 3.0 `
  --maximum-duration 60 `
  --wasapi-exclusive `
  --pressure-output `
  --pressure-pa-per-full-scale 20 `
  --output-peak-limit 0.1
```

Diese Abbildung ist ein technischer Hoermonitor und keine kalibrierte
Wiedergabe des simulierten Schalldruckpegels. Die Metadaten protokollieren
Skalierung, Grenzwert, erreichten Digitalpeak, begrenzte Samples und nicht
endliche Drucksamples. Der Rohdruck in `output_pressure_pa.npy` wird durch die
Monitorstufe nicht veraendert.

## Ablauf und Ergebnisdateien

Vor Streambeginn werden WAV, Spannungseingang, Druckausgang und alle
Metrikarrays im RAM vorbereitet. PortAudio ruft danach pro Hardwareblock genau
einmal `process_callback()` auf. Im Callback finden weder GUI-Aufrufe noch
Dateizugriffe statt. Erst nach dem letzten Block werden unter
`data/AudioCallbackTests` geschrieben:

- `output_pressure_pa.npy`: wissenschaftlicher Modelldruck in Pascal,
- `callback_timing.csv`: Blockzeiten und Treiberstatus je Block,
- `run_metadata.json`: Konfiguration, Audiohardware, Softwareversionen und
  zusammengefasste Deadline- und Ausgangsschutzmetriken.

Ein Treiber-Underflow ist dabei wichtiger als eine nur intern gemessene
Zeitueberschreitung: Er zeigt an, dass PortAudio die Ausgabe tatsaechlich nicht
rechtzeitig versorgen konnte. Fuer eine belastbare Aussage sind mehrere lange
Laeufe mit 44,1 und 48 kHz sowie 128 und 256 Samples sinnvoll.

Die GUI ist in diesem Schritt absichtlich nicht beteiligt. Sie kann spaeter nur
Start, Abbruch und Anzeige steuern, ohne selbst im Audiocallback zu rechnen.
