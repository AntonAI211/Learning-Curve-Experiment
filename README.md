# Lernkurven-Experiment

Experimenteller Teil der Facharbeit *"Lernkurven in der KI: Eine mathematische Analyse von Sättigung und abnehmenden Grenzerträgen"*.

Dieses Projekt trainiert ein neuronales Netzwerk auf CIFAR-10 mit variierenden Trainingsdatenmengen, um empirische Lernkurven zu erzeugen. Anschließend wird das Power-Law-Modell

```
A(n) = A_inf + eta * n^gamma
```

an die Messdaten gefittet, um Sättigung und abnehmende Grenzerträge quantitativ nachzuweisen.

---

## Inhaltsverzeichnis

1. [Voraussetzungen](#1-voraussetzungen)
2. [Installation](#2-installation)
3. [Projektstruktur](#3-projektstruktur)
4. [Konfiguration](#4-konfiguration)
5. [Experiment durchführen](#5-experiment-durchführen)
6. [Power-Law-Fit und Auswertung](#6-power-law-fit-und-auswertung)
7. [Plots erstellen](#7-plots-erstellen)
8. [Ergebnisdateien](#8-ergebnisdateien)
9. [Mathematischer Hintergrund](#9-mathematischer-hintergrund)
10. [Modellarchitekturen](#10-modellarchitekturen)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Voraussetzungen

- **Hardware:** MacBook Pro M4 Pro (oder anderer Mac mit Apple Silicon)
- **Python:** 3.9+
- **Betriebssystem:** macOS (MPS-Backend wird automatisch erkannt)

### Abhängigkeiten

| Paket        | Zweck                                    |
|-------------|------------------------------------------|
| `torch`      | Deep-Learning-Framework (MPS-Backend)    |
| `torchvision`| CIFAR-10 Dataset und Bildtransformationen|
| `numpy`      | Numerische Berechnungen                  |
| `pandas`     | Datenhandling und CSV-Export             |
| `scipy`      | Power-Law Curve-Fitting                  |
| `matplotlib` | Visualisierung / Plots                   |
| `tqdm`       | Fortschrittsanzeige                      |

---

## 2. Installation

```bash
# In das Projektverzeichnis wechseln
cd learning_curve_experiment

# Alle Abhängigkeiten installieren
python3 -m pip install torch torchvision numpy pandas scipy matplotlib tqdm
```

### MPS-Verfügbarkeit prüfen

```bash
python3 -c "import torch; print(f'MPS verfügbar: {torch.backends.mps.is_available()}')"
```

Erwartete Ausgabe: `MPS verfügbar: True`

Falls `False`: Stelle sicher, dass PyTorch >= 2.0 installiert ist (`python3 -c "import torch; print(torch.__version__)"`).

---

## 3. Projektstruktur

```
learning_curve_experiment/
├── config.py              # Alle Hyperparameter zentral an einem Ort
├── models.py              # SimpleCNN (~621k Params) + ResNet18 (~11.2M Params)
├── dataset.py             # CIFAR-10 Laden, Train/Val-Split, Subset-Erzeugung
├── train.py               # Trainingsloop mit Early Stopping + LR-Scheduling
├── experiment.py          # Hauptschleife über alle (n, k)-Kombinationen
├── fit_power_law.py       # Power-Law-Fit mit scipy.optimize.curve_fit
├── visualize.py           # Alle Plots (Lernkurve, Residuen, Ableitungen, ...)
├── requirements.txt       # Python-Abhängigkeiten
├── results/               # Ergebnisse (JSON + CSV) - wird automatisch erstellt
│   ├── individual_runs.json
│   └── aggregated_results.csv
├── plots/                 # Gespeicherte Abbildungen - wird automatisch erstellt
│   ├── lernkurve_power_law_fit.png
│   ├── residuenplot.png
│   ├── trainingskurven.png
│   └── grenzertrag_ableitung.png
└── data/                  # CIFAR-10 Download - wird automatisch erstellt
```

### Welche Datei macht was?

| Datei               | Beschreibung |
|---------------------|-------------|
| `config.py`         | Zentrale Konfiguration. Hier werden Modell, Optimizer, Trainingsgrößen, Seed-Logik und Fit-Parameter definiert. |
| `models.py`         | Definiert zwei Modelle: `SimpleCNN` (3 Conv-Blöcke + 2 FC-Layer) und `ResNet18CIFAR` (angepasstes ResNet-18 für 32x32-Bilder). |
| `dataset.py`        | Lädt CIFAR-10, teilt in Train-Pool/Validation, erzeugt Subsets der Größe n für jede Partition k. |
| `train.py`          | Trainiert ein Modell mit Early Stopping (Patience 10) und ReduceLROnPlateau. Gibt Metriken als Dictionary zurück. |
| `experiment.py`     | Orchestriert das gesamte Experiment: iteriert über alle Trainingsgrößen und Partitionen, speichert Ergebnisse nach jedem Run. |
| `fit_power_law.py`  | Fittet `A(n) = A_inf + eta * n^gamma` an die aggregierten Ergebnisse. Berechnet R², Residuen und Standardfehler. |
| `visualize.py`      | Erstellt vier Plots: Lernkurve mit Fit, Residuen, Trainingskurven pro Epoche, erste Ableitung (Grenzerträge). |

---

## 4. Konfiguration

Alle Einstellungen befinden sich in `config.py`. Die wichtigsten Parameter:

### Modell und Optimizer wählen

```python
# In config.py ändern:
MODEL_NAME = "SimpleCNN"   # oder "ResNet18"
OPTIMIZER_NAME = "Adam"    # oder "SGD"
```

| Einstellung       | SimpleCNN                  | ResNet18                          |
|-------------------|----------------------------|-----------------------------------|
| Parameter         | ~621.000                   | ~11.200.000                       |
| Architektur       | 3 Conv + 2 FC (from scratch)| torchvision ResNet-18 (angepasst)|
| Geschätzte Dauer  | ~1-3 Stunden               | ~6-12 Stunden                     |
| Empfehlung        | Für die Facharbeit         | Ambitionierter, näher an Literatur|

### Trainingsgrößen

```python
TRAINING_SIZES = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 48000]
```

Logarithmisch verteilt, um Power-Law-Verhalten über mehrere Größenordnungen (10^2 bis ~5*10^4) sichtbar zu machen.

### Partitionen (Wiederholungen)

```python
NUM_PARTITIONS = 5  # k=5 verschiedene zufällige Subsets pro n
```

Für jede Trainingsgröße n werden 5 verschiedene zufällige Subsets gezogen und unabhängig trainiert. Ausnahme: n=48.000 (gesamter Pool) wird nur 1x trainiert. **Insgesamt: 8 x 5 + 1 = 41 Runs.**

### Trainings-Hyperparameter

| Parameter               | Wert          | Beschreibung                                    |
|-------------------------|---------------|-------------------------------------------------|
| `BATCH_SIZE`            | 128           | Mini-Batch-Größe                                |
| `MAX_EPOCHS`            | 100           | Maximale Epochen pro Run                        |
| `EARLY_STOPPING_PATIENCE`| 10           | Stoppt, wenn Val-Loss 10 Epochen nicht sinkt    |
| `LR_SCHEDULER_FACTOR`   | 0.5          | LR wird halbiert bei Plateau                    |
| `LR_SCHEDULER_PATIENCE`  | 5           | Plateau-Erkennung nach 5 Epochen                |
| `ADAM_LR`               | 0.001         | Learning Rate für Adam                          |
| `SGD_LR`                | 0.1           | Learning Rate für SGD                           |
| `DROPOUT_CONV`          | 0.25          | Dropout nach Conv-Blöcken                       |
| `DROPOUT_FC`            | 0.5           | Dropout nach FC-Layer                           |

### Data Augmentation

Bewusst minimal gehalten, um die Lernkurve nicht zu verzerren:
- `RandomHorizontalFlip` (50% Wahrscheinlichkeit)
- `RandomCrop(32, padding=4)` (leichtes Verschieben)
- `Normalize` mit CIFAR-10-Mittelwert und Standardabweichung

---

## 5. Experiment durchführen

### Gesamtes Experiment starten

```bash
cd learning_curve_experiment
python3 experiment.py
```

Das Skript macht Folgendes:
1. Lädt CIFAR-10 herunter (beim ersten Mal, ~170 MB)
2. Teilt die 50.000 Trainingsbilder in **48.000 Training-Pool** + **2.000 Validation**
3. Iteriert über alle Trainingsgrößen n = {100, 250, ..., 48000}
4. Für jedes n: zieht k=5 zufällige Subsets und trainiert je ein Modell
5. Speichert **nach jedem Run** sofort in `results/individual_runs.json`
6. Am Ende: aggregiert alle Ergebnisse in `results/aggregated_results.csv`

### Konsolenausgabe während des Trainings

```
Lade CIFAR-10...
Train-Pool: 48000, Val: 2000, Test: 10000
Modell: SimpleCNN (620,810 Parameter)
Optimizer: Adam
Device: mps

Starte Experiment: 41 Runs total, 0 bereits abgeschlossen.

[n=  100, k=0] Training startet (seed=10000)...
[n=  100, k=0] Test-Acc: 0.2854 | Val-Acc: 0.2950 | Epochen: 34 | Zeit: 42.1s | (1/41)
[n=  100, k=1] Training startet (seed=10001)...
...
```

### Checkpoint-System (Fortsetzen nach Abbruch)

Das Experiment speichert nach **jedem einzelnen Run** die Ergebnisse. Wenn das Skript unterbrochen wird (Ctrl+C, Laptop zugeklappt, etc.), kann es einfach neu gestartet werden:

```bash
python3 experiment.py   # Erkennt automatisch bereits abgeschlossene Runs
```

Ausgabe: `Fortgesetzt: 12 vorhandene Runs gefunden.`

Bereits abgeschlossene (n, k)-Kombinationen werden übersprungen.

### Datenaufteilung im Detail

```
CIFAR-10 (50.000 Trainingsbilder + 10.000 Testbilder)
│
├── 48.000 Training-Pool
│   ├── Subset n=100   (5x zufällig gezogen, je unterschiedlicher Seed)
│   ├── Subset n=250   (5x)
│   ├── Subset n=500   (5x)
│   ├── ...
│   ├── Subset n=25000 (5x)
│   └── Subset n=48000 (1x, gesamter Pool)
│
├── 2.000 Validation (fester Split, für Early Stopping + LR-Scheduling)
│
└── 10.000 Test (original, unveränderlich, für finale Accuracy-Messung)
```

### Reproduzierbarkeit

Jeder Run verwendet einen deterministischen Seed: `seed = n * 100 + k`

Beispiele:
- n=1000, k=0 → seed = 100000
- n=1000, k=3 → seed = 100003
- n=5000, k=2 → seed = 500002

Dadurch sind alle Ergebnisse exakt reproduzierbar.

---

## 6. Power-Law-Fit und Auswertung

### Nur den Fit ausführen (ohne Plots)

```bash
python3 fit_power_law.py
```

Voraussetzung: `results/aggregated_results.csv` muss existieren (wird von `experiment.py` erstellt).

### Ausgabe

```
============================================================
Power-Law-Fit: A(n) = A_inf + eta * n^gamma
============================================================
  A_inf (Asymptote)  = 0.932145  (± 0.008234)
  eta                = -3.456789  (± 0.234567)
  gamma              = -0.287654  (± 0.012345)
  R²                 = 0.998765
============================================================

  → Maximale Genauigkeit:  93.21%
  → Sättigungsexponent:    -0.2877

         n  A(n) beob.  A(n) pred.    Residuum
  ────────  ──────────  ──────────  ──────────
       100      0.2854      0.2812     +0.0042
       250      0.3721      0.3698     +0.0023
       ...
     48000      0.9198      0.9204     -0.0006
```

### Fit-Parameter im Detail

| Parameter | Bedeutung | Erwarteter Bereich |
|-----------|-----------|-------------------|
| `A_inf`   | Asymptotische maximale Accuracy. Der Grenzwert, gegen den A(n) für n→∞ konvergiert. | 0.85 - 0.95 (CIFAR-10, SimpleCNN) |
| `eta`     | Skalierungsfaktor. Negativ, da A(n) von unten gegen A_inf konvergiert. | < 0 |
| `gamma`   | Sättigungsexponent. Bestimmt, wie schnell die Sättigung eintritt. | -0.35 bis -0.07 (laut Literatur) |
| `R²`      | Bestimmtheitsmaß. Misst, wie gut das Modell die Daten beschreibt. | Sollte > 0.99 sein |

### Methodik des Fits

- **Algorithmus:** `scipy.optimize.curve_fit` (Levenberg-Marquardt, nichtlineare Regression)
- **Gewichtung:** Standardabweichung als Sigma (Runs mit kleiner Streuung werden stärker gewichtet)
- **Startwerte:** A_inf=0.9, eta=-10, gamma=-0.5
- **Bounds:** A_inf ∈ (0, 1), eta ∈ (-∞, 0), gamma ∈ (-1, 0)

---

## 7. Plots erstellen

### Alle Plots auf einmal

```bash
python3 visualize.py
```

Erstellt vier Plots in `plots/`:

### Plot 1: Lernkurve mit Power-Law-Fit (`lernkurve_power_law_fit.png`)

**Der Hauptplot der Facharbeit.**
- x-Achse: Trainingsdatenmenge n (logarithmisch skaliert)
- y-Achse: Accuracy A(n) (0 bis 1)
- Blaue Punkte: Gemessene mittlere Accuracy mit Fehlerbalken (Standardabweichung)
- Rote Kurve: Gefittetes Power-Law-Modell A(n) = A_inf + eta * n^gamma
- Graue gestrichelte Linie: Asymptote A_inf
- Legende mit exakten Parametern und R²

### Plot 2: Residuenplot (`residuenplot.png`)

Zeigt die Abweichung zwischen gemessenen und vorhergesagten Werten.
- Balkendiagramm: Residuum = A_beobachtet - A_vorhergesagt
- Idealfall: Alle Balken nahe Null, keine systematische Abweichung

### Plot 3: Trainingskurven (`trainingskurven.png`)

Validation Accuracy und Validation Loss pro Epoche für ausgewählte Trainingsgrößen (n=100, 500, 2500, 10000, 48000).
- Zeigt, wie bei kleinem n schneller konvergiert (bzw. overfittet) wird
- Zeigt den Einfluss der Datenmenge auf den Trainingsverlauf

### Plot 4: Grenzertrag / Erste Ableitung (`grenzertrag_ableitung.png`)

Visualisiert A'(n) = eta * gamma * n^(gamma-1):
- x-Achse: n (logarithmisch)
- y-Achse: A'(n) (Accuracy-Gewinn pro zusätzlichem Trainingsbeispiel)
- Zeigt die abnehmenden Grenzerträge: Die Kurve fällt monoton gegen Null

---

## 8. Ergebnisdateien

### `results/individual_runs.json`

Enthält **alle Einzelergebnisse** als JSON-Array. Jeder Eintrag hat folgende Felder:

```json
{
  "training_size": 1000,
  "partition_index": 2,
  "seed": 100002,
  "model": "SimpleCNN",
  "optimizer": "Adam",
  "final_test_accuracy": 0.5234,
  "final_test_loss": 1.3456,
  "best_val_accuracy": 0.5350,
  "best_val_loss": 1.3012,
  "epochs_trained": 47,
  "training_time_seconds": 85.23,
  "train_loss_per_epoch": [2.30, 1.98, ...],
  "train_accuracy_per_epoch": [0.10, 0.18, ...],
  "val_loss_per_epoch": [2.28, 1.95, ...],
  "val_accuracy_per_epoch": [0.11, 0.19, ...]
}
```

| Feld                      | Beschreibung                                           |
|---------------------------|--------------------------------------------------------|
| `training_size`           | Anzahl Trainingsbeispiele n                            |
| `partition_index`         | Partition k (0 bis 4)                                  |
| `seed`                    | Verwendeter Random Seed                                |
| `model`                   | Modellname ("SimpleCNN" oder "ResNet18")               |
| `optimizer`               | Optimizer ("Adam" oder "SGD")                          |
| `final_test_accuracy`     | **Accuracy auf dem Testset (der zentrale Messwert)**   |
| `final_test_loss`         | Loss auf dem Testset                                   |
| `best_val_accuracy`       | Beste Validation Accuracy (beim besten Val-Loss)       |
| `best_val_loss`           | Bester Validation Loss (Kriterium für Early Stopping)  |
| `epochs_trained`          | Anzahl tatsächlich trainierter Epochen                 |
| `training_time_seconds`   | Trainingszeit in Sekunden                              |
| `train_loss_per_epoch`    | Training Loss pro Epoche (für Trainingskurven-Plot)    |
| `train_accuracy_per_epoch`| Training Accuracy pro Epoche                           |
| `val_loss_per_epoch`      | Validation Loss pro Epoche                             |
| `val_accuracy_per_epoch`  | Validation Accuracy pro Epoche                         |

### `results/aggregated_results.csv`

Aggregierte Ergebnisse pro Trainingsgröße (Eingabe für den Power-Law-Fit):

```csv
n,mean_accuracy,std_accuracy,min_accuracy,max_accuracy,num_runs
100,0.2854,0.0123,0.2701,0.2998,5
250,0.3721,0.0098,0.3598,0.3856,5
...
48000,0.9198,0.0000,0.9198,0.9198,1
```

| Spalte           | Beschreibung                                                |
|------------------|-------------------------------------------------------------|
| `n`              | Trainingsgröße                                              |
| `mean_accuracy`  | Mittelwert der Test-Accuracy über alle k Partitionen        |
| `std_accuracy`   | Standardabweichung (für Fehlerbalken in Plots)              |
| `min_accuracy`   | Minimum über alle Partitionen                               |
| `max_accuracy`   | Maximum über alle Partitionen                               |
| `num_runs`       | Anzahl durchgeführter Runs (5 oder 1 für n=48000)           |

---

## 9. Mathematischer Hintergrund

### Das Power-Law-Modell

```
A(n) = A_inf + eta * n^gamma
```

- **A(n):** Genauigkeit (Accuracy) auf Testdaten, gemessen an einem festen Testsatz
- **n:** Anzahl der Trainingsbeispiele
- **A_inf > 0:** Asymptotische maximale Genauigkeit (Grenzwert für n → ∞)
- **eta < 0:** Skalierungsfaktor (negativ, da A(n) < A_inf)
- **gamma < 0:** Sättigungsexponent (negativ, damit n^gamma → 0 für n → ∞)

### Erste Ableitung (Grenzerträge)

```
A'(n) = eta * gamma * n^(gamma - 1)
```

Da eta < 0 und gamma < 0, ist eta * gamma > 0, also A'(n) > 0 für alle n > 0.
Die Accuracy steigt monoton mit wachsender Datenmenge.

Grenzverhalten: lim(n→∞) A'(n) = 0 (Grenzerträge gehen gegen Null).

### Zweite Ableitung (abnehmende Grenzerträge)

```
A''(n) = eta * gamma * (gamma - 1) * n^(gamma - 2)
```

Es gilt A''(n) < 0 für alle n (Konkavität), was mathematisch beweist, dass die Grenzerträge monoton abnehmen.

### Bezug zur Facharbeit

Die empirischen Daten dieses Experiments sollen zeigen:
1. Das Power-Law-Modell beschreibt die Lernkurve gut (R² nahe 1)
2. Die Sättigung ist quantifizierbar (A_inf als Grenzwert)
3. Die abnehmenden Grenzerträge sind nachweisbar (gamma < 0)

---

## 10. Modellarchitekturen

### SimpleCNN (~621.000 Parameter)

```
Input: 3x32x32 (RGB CIFAR-10 Bild)
  │
  ├── Conv2d(3→32, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.25)
  │   Ausgabe: 32x16x16
  │
  ├── Conv2d(32→64, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.25)
  │   Ausgabe: 64x8x8
  │
  ├── Conv2d(64→128, 3x3) → BatchNorm → ReLU → MaxPool(2x2) → Dropout(0.25)
  │   Ausgabe: 128x4x4
  │
  ├── Flatten → Linear(2048→256) → ReLU → Dropout(0.5)
  │
  └── Linear(256→10) → Output (10 Klassen)
```

### ResNet18CIFAR (~11.200.000 Parameter)

Basierend auf `torchvision.models.resnet18`, angepasst für 32x32-Bilder:
- Erste Conv-Schicht: 3x3 (statt 7x7 für ImageNet)
- Stride=1 (statt 2, da Bilder klein sind)
- MaxPool nach erster Conv entfernt
- FC-Layer: 512 → 10 Klassen
- **Kein Pretraining** (`weights=None`)

---

## 11. Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
python3 -m pip install torch torchvision
```

### "MPS verfügbar: False"

- Prüfe PyTorch-Version: `python3 -c "import torch; print(torch.__version__)"`
- MPS benötigt PyTorch >= 2.0 und macOS 12.3+
- Fallback: Das Experiment läuft auch auf CPU (automatisch), nur langsamer

### Training ist sehr langsam

- Prüfe, ob MPS aktiv ist (Konsolenausgabe zeigt `Device: mps`)
- Die ersten Epochen sind auf MPS wegen JIT-Kompilierung langsamer
- Für schnellere Tests: In `config.py` die Trainingsgrößen reduzieren:
  ```python
  TRAINING_SIZES = [100, 500, 5000, 48000]
  NUM_PARTITIONS = 3
  ```

### Experiment abgebrochen, Ergebnisse verloren?

Nein. Das Checkpoint-System speichert nach jedem einzelnen Run. Einfach `python3 experiment.py` erneut starten.

### Ergebnisse zurücksetzen (komplett neu starten)

```bash
rm results/individual_runs.json results/aggregated_results.csv
python3 experiment.py
```

### Power-Law-Fit schlägt fehl

- Prüfe, ob `results/aggregated_results.csv` existiert und Werte enthält
- Falls der Fit nicht konvergiert: Startwerte in `config.py` anpassen:
  ```python
  FIT_P0 = [0.9, -10.0, -0.5]  # [A_inf, eta, gamma]
  ```
- Das Skript versucht automatisch einen zweiten Fit ohne Gewichtung