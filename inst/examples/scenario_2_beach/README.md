# Scenario 2: Coastal Beach Habitat (Ashkelon)

Local microclimate correction for a single logger at a coastal beach site.
Conditions are strongly influenced by marine winds and sea temperatures,
making this a more challenging correction target than inland habitats.

## Run

```r
source(system.file("examples", "scenario_2_beach", "run_scenario_2.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Beach_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for the Ashkelon beach site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | CSV with RMSE and R² before and after correction |
| `prediction_examples_beach.png` | 120-hour observed vs. corrected predictions |
| `learning_curves_beach.png` | RMSE as a function of training data size |
| `scenario_2_report.md` | Full results report |

## Key result

RMSE reduced from ~8.2 °C (NicheMapR baseline) to ~3.1 °C (RF) and ~4.0 °C (LSTM).
At least 28 days of training data are needed to capture coastal wind and tide dynamics.
