# Scenario 1: Mediterranean Valley Habitat (Harod)

Local microclimate correction for a single logger in a Mediterranean valley.
A Random Forest and an LSTM model are each trained on logger data from the Harod site
and used to correct NicheMapR temperature predictions.

## Run

```r
source(system.file("examples", "scenario_1_valley", "run_scenario_1.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Harod_dataset.csv` | Pre-aligned logger + NicheMapR data for the Harod valley site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | CSV with RMSE and R² before and after correction |
| `prediction_examples_valley.png` | 120-hour observed vs. corrected predictions |
| `learning_curves_valley.png` | RMSE as a function of training data size |
| `scenario_1_report.md` | Full results report |

## Key result

RMSE reduced from ~7.3 °C (NicheMapR baseline) to ~2.3 °C (LSTM) and ~2.7 °C (RF),
an improvement of roughly 58–61%.
