# Hardwareunabhaengiger Audiocallback-Kern

`src/RealtimeAudioEngine.py` ist die vorbereitende Stufe fuer einen spaeteren
PortAudio- beziehungsweise `sounddevice`-Callback. Sie ist unabhaengig von
Tkinter und besitzt noch keine reale Audiohardwareanbindung.

## Aktueller Umfang

- 44,1 und 48 kHz,
- zustandsbehaftete Verarbeitung mit dem Fast Solver,
- feste Blockgroesse,
- vollstaendig vor dem Stream geladene und auf Spannung skalierte WAV,
- stummer oder abgesenkter trockener Audioausgang,
- vorallokierter Float64-Druckausgang,
- Callback-, Solver-, Threadarbeits- und Treiberstatusmetriken,
- Deadline-Episoden und p99,9,
- hardwarefreier simulierter Audiotreiber fuer Regressionstests.

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
abgesenkte Eingangssignal. Der berechnete Druck in Pascal wird nicht als
Audiospannung interpretiert.

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

## Naechster Schritt

Ein spaeterer Adapter fuer PortAudio beziehungsweise `sounddevice` uebernimmt:

1. Auswahl und Pruefung des Ausgabegeraets,
2. Abbildung der Treiberstatus auf `AudioCallbackStatus`,
3. Start und Stop des Streams,
4. Weitergabe des Treiberausgangspuffers an `process_callback()`,
5. Speicherung der Ergebnisse erst nach Ende des Streams.

Die GUI wird danach nur Controller und Anzeige. Sie ruft den Audiocallback
nicht selbst auf.
