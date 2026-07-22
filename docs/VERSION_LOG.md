# Versionslog – Simulator und wissenschaftliche Auswertung

## Arbeitsstand 2026-07-16

### Numerischer Simulator

- Physikalisch positive Modellparameter und zeitvariante WDF-Portwiderstände werden explizit validiert. Ungültige negative Werte von $L_e(x)$, $K_{ms}(x)$ oder den daraus abgeleiteten Portwiderständen werden nicht mehr still auf eine positive Untergrenze geklemmt, sondern führen zu einer diagnostischen Fehlermeldung.
- Die Paper-Presets verwenden für die automatische Frequenzwahl die aus dem Maximum von $|Z_e(f)|$ bestimmten Bernardini-Resonanzen $f_r=57{,}5\,\mathrm{Hz}$ (Spk-1) und $f_r=162{,}8\,\mathrm{Hz}$ (Spk-2). Für freie Parametersätze wird das Maximum der linearen Ersatzschaltungsimpedanz numerisch bestimmt.
- `simulate_model()` um einen optionalen Analysemodus (`record_traces=True`) erweitert. Neben dem Ausgangsdruck werden Eingangsspannung, Membranauslenkung, Membrangeschwindigkeit, Volumenstrom und Schwingspulenstrom sampleweise aufgezeichnet.
- Der Standardpfad liefert weiterhin ausschließlich $P_\mathrm{out}(t)$. Die Gleichheit des Druckausgangs beider Pfade wurde samplegenau mit `np.array_equal` verifiziert.
- WDF-Streugleichungen, nichtlineare Kennlinien, Lautsprecherparameter, Zustandsaktualisierung und Portadaption blieben unverändert. Die Erweiterung beobachtet ausschließlich bereits berechnete Zustandsgrößen.

### Stationäre Spektralanalyse

- Einseitige FFT-Normierung korrigiert: Positive Frequenzen werden als Spitzenamplituden rekonstruiert; DC und Nyquist werden nicht mehr verdoppelt.
- Mittlere Leistung eines kohärenten Sinustons auf
  \[
  P_k=\frac{A_{\mathrm{peak},k}^{2}}{2}
  \]
  normiert. Die frühere Auswertung verwendete $A_\mathrm{peak}^{2}$ und lag damit um $3{,}0103\,\mathrm{dB}$ zu hoch.
- Relative Harmonischenpegel und THD bleiben durch die gemeinsame Skalierung unverändert; absolute Leistungsspektren sind nun physikalisch konsistent.
- Harmonische oberhalb Nyquist werden nicht mehr dem letzten FFT-Bin zugeordnet, sondern gemäß
  \[
  k_\mathrm{max}=\left\lfloor\frac{F_s/2}{f_\mathrm{in}}\right\rfloor
  \]
  aus der Auswertung ausgeschlossen.

### Bernardini-Messdaten und Referenzvergleich

- Die Anregung unterscheidet explizit zwischen einem physikalischen RMS-Nennpegel und dem empirischen Bernardini-Reproduktionsfaktor. Im Reproduktionsmodus gilt $V_{\mathrm{RMS,eff}}=\sqrt{2}\,A_\mathrm{in}$; der unveränderte Nennwert $A_\mathrm{in}$ bleibt für die Zuordnung der Referenzspalte maßgeblich. Standardmäßig bleibt der Skalierungsfaktor eins.
- Solver-unabhängige Referenz- und Vergleichsschicht in `spectral_comparison.py` eingeführt. Messgröße, Einheit, Skalierung, Frequenzraster, Versuchskonfiguration sowie bestätigte und angenommene PSD-Parameter werden explizit geführt.
- Fig. 4 wird als $|Z_e(f)|$ in $\Omega$ beziehungsweise $|P_\mathrm{out}(f)|$ in Pa interpretiert. Eine dritte Spalte wird nur ignoriert, wenn sie vollständig null ist.
- Zeitdaten werden mit $F_s=96\,\mathrm{kHz}$ interpretiert. Mehrfachspalten werden eindeutig über Paperfrequenz und RMS-Pegel ausgewählt; eine lediglich „nächstliegende“ Konfiguration ist nicht zulässig.
- Zweitonvergleiche validieren zusätzlich
  \[
  \varphi_1=0{,}8472\,\mathrm{rad},\qquad \varphi_2=2{,}2179\,\mathrm{rad}.
  \]
- Messung und Simulation werden ohne Amplitudenskalierung durch Kreuzkorrelation zeitlich ausgerichtet. Kompensiert wird ausschließlich eine unbekannte Zeitverschiebung.
- Für Zeitbereichsreferenzen werden RMSE in Pa und der relative RMS-NRMSE
  \[
  \mathrm{NRMSE}=\frac{\lVert P_\mathrm{sim}-P_\mathrm{meas}\rVert_2}{\lVert P_\mathrm{meas}\rVert_2}
  \]
  über den ausgerichteten Ausschnitt ausgegeben. Samplezahl, Dauer und Startindex des Simulationsausschnitts werden mitgeführt; eine Amplitudennormierung findet nicht statt.
- PSD-Referenzen aus Fig. 6(c/d), 7(c/d) und 8(c/d) werden, sofern vorhanden, automatisch mit den zugehörigen Zeitmessungen aus Fig. 6(a/b), 7(a/b) beziehungsweise 8(a/b) verknüpft. Dadurch bleiben Spektralvergleich und Zeitbereichs-NRMSE derselben Simulation gemeinsam verfügbar.
- PSD-Vergleiche verlangen übereinstimmende Abtastrate, Preset, RMS-Pegel, Frequenzen und gegebenenfalls Phasen und verwenden exakt das Frequenzraster der Messdatei.
- Der DC-Anteil der Simulation bleibt für Referenz-PSD-Vergleiche erhalten; ausgewertet wird der unveränderte Ausgang nach der gewählten Einschwingzeit.

### PSD-Normierung und Analyseparameter

- Fig. 6(c/d) ist gegenüber $\mathrm{Pa^2/Hz}$ um annähernd $10^6$ skaliert. Ausschließlich dort wird angewendet:
  \[
  S_{PP,\mathrm{Pa^2/Hz}}=10^{-6}S_{PP,\mathrm{Datei}}.
  \]
- Parseval-Verhältnisse nach Korrektur: Fig. 6(c) $0{,}99339$, Fig. 6(d) $0{,}98809$. Fig. 7(c/d) benötigt keine Korrektur und ergibt $1{,}00364$ beziehungsweise $0{,}99136$. Fig. 8(c/d) wird ebenfalls ohne Faktor $10^{-6}$ interpretiert.
- Darstellung linearer Leistungsdichten über
  \[
  S_{PP,\mathrm{dB/Hz}}=10\log_{10}\!\left(S_{PP,\mathrm{linear}}\right).
  \]
- Aus $F_s=96\,\mathrm{kHz}$ und den Messrastern folgen:
  - Fig. 6/7: $\Delta f=5\,\mathrm{Hz}$ und $N_\mathrm{FFT}=19200$,
  - Fig. 8: $\Delta f\approx0{,}83333\,\mathrm{Hz}$ und $N_\mathrm{FFT}=115200$.
- Hamming-Fenster und Segmentlänge gleich FFT-Länge wurden über Leakage-Verhältnisse mehrerer Grundtöne in Fig. 6(c), 7(c) und 7(d) bestätigt.
- Nicht eindeutig bestimmbar bleiben Segmentüberlappung und Detrending. Bis zum Vorliegen des Originalcodes werden $50\,\%$ Überlappung und kein Detrending ausdrücklich als Annahmen geführt.

### Verifikation

- Alle 16 bereitgestellten Messdateien klassifiziert und validiert.
- Parseval-, Raster-, Einheiten-, Parameter-, Phasen- und Nyquist-Prüfungen automatisiert ausgeführt.
- Änderungen der Auswertung getrennt vom unveränderten WDF-Druckausgang geprüft.

## Arbeitsstand 2026-07-22

### Zeitbereichsausrichtung und NRMSE

- Die Lag-Suche verwendet eine normierte Kreuzkorrelation. Jedes mögliche
  Simulationsfenster wird mit seiner eigenen mittelwertfreien Energie
  normiert; ein lediglich energiereicherer Ausschnitt kann dadurch nicht mehr
  gegenüber einem formähnlicheren Ausschnitt bevorzugt werden.
- Nach der ganzzahligen Lag-Suche wird ein lokaler Subsample-Zeitversatz durch
  kubische Interpolation bestimmt. Optimiert wird ausschließlich der
  mittelwertfreie L2-Fehler; eine multiplikative Pegelanpassung findet weiterhin
  nicht statt.
- Der absolute NRMSE bleibt als
  \[
  \mathrm{NRMSE}_{\mathrm{abs}}=
  \frac{\lVert P_\mathrm{sim}-P_\mathrm{meas}\rVert_2}
       {\lVert P_\mathrm{meas}\rVert_2}
  \]
  erhalten. Zusätzlich wird der offsetunabhängige AC-NRMSE ausgewiesen:
  \[
  \mathrm{NRMSE}_{\mathrm{AC}}=
  \frac{\lVert(P_\mathrm{sim}-\overline P_\mathrm{sim})-
                   (P_\mathrm{meas}-\overline P_\mathrm{meas})\rVert_2}
       {\lVert P_\mathrm{meas}-\overline P_\mathrm{meas}\rVert_2}.
  \]
- Ganzzahliger Startindex, Subsample-Offset, resultierender Zeitversatz und
  Korrelationskoeffizient werden gemeinsam mit beiden Fehlermaßen gespeichert.
- Verifikation am Spk-1-Zweitonfall bei 11,5 V: AC-NRMSE 1,526 %, gegenüber
  1,5543 % in Bernardini et al., Tabelle 5. Der absolute NRMSE beträgt nach der
  verbesserten zeitlichen Ausrichtung 4,728 %.
