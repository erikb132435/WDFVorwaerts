# Wissenschaftliches Validierungspaket

Die GUI-Funktion `Validierungspaket speichern` archiviert einen einzelnen,
bereits ausgefuehrten Vergleich zwischen Fast Solver und Bernardini-Messdaten.
Sie fuehrt keine zweite Simulation und keine abweichende Auswertung durch.

Der Export ist nur verfuegbar, wenn der schnelle Solver die Primaerausgabe
geliefert hat und mindestens ein Zeit- oder PSD-Messvergleich berechnet wurde.
Der beim Lauf verwendete Konfigurationsstand wird unabhaengig von spaeteren
GUI-Aenderungen gespeichert.

## Ordnerinhalt

- `metadata.json`: Fallkennung, Laufkonfiguration, Modellparameter,
  Softwareversionen, Referenzmetadaten, Definitionen, Kennwerte und Dateiliste.
- `metrics.csv`: skalare Zeit- und Spektralkennwerte mit Einheit, Bereich und
  Definition.
- `simulation_raw.npz`: alle numerischen Arrays des Simulationsergebnisses.
- `time_comparison.csv`: Zeit, Messdruck, zeitlich ausgerichteter
  Simulationsdruck und Differenz in Pascal.
- `spectrum_comparison.csv`: Messung und Simulation auf dem Frequenzraster der
  Referenzdatei, jeweils linear und in dB, sowie dB-Differenz und relevante
  Frequenzmaske.
- `references/`: unveraenderte Kopien der ausgewaehlten Referenz und einer
  gegebenenfalls automatisch zugeordneten Zeitreferenz. SHA-256-Pruefsummen
  stehen in `metadata.json`.
- `comparison_plot.pdf`: die beim Export aktuell angezeigte GUI-Darstellung.

Nicht verfuegbare Vergleichsarten werden nicht durch leere oder synthetische
Dateien ersetzt. Bei einer reinen Zeitreferenz entfaellt daher beispielsweise
`spectrum_comparison.csv`.

## Ergebnisbewertung

Das Paket verwendet den Status `evaluated_without_acceptance_threshold`. Es
wendet bewusst keine automatische Gut-/Schlecht-Grenze an. AC-NRMSE, absoluter
NRMSE und spektrale dB-Fehler werden als wissenschaftliche Beobachtungen
gespeichert; ihre anwendungsbezogene Bewertung bleibt davon getrennt.

Der spektrale RMSE wird nur auf endlichen Punkten berechnet, deren
Referenzleistung innerhalb des dokumentierten relevanten Dynamikbereichs liegt.
Die Originalkurven bleiben unabhaengig davon vollstaendig erhalten.
