# Scenario 1: Mediterranean Valley Habitat (Harod)

Local microclimate correction for a single logger in a Mediterranean valley.
A Random Forest and an LSTM model are each trained on logger data from the Harod site
and used to correct NicheMapR temperature predictions.

## Run

```r
source(system.file("examples", "scenario_1_valley_single_logger", "run_scenario_1.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Harod_dataset.csv` | Pre-aligned logger + NicheMapR data for the Harod valley site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | Per-microhabitat CSV with RMSE before and after correction |
| `prediction_examples_valley.png` | 120-hour observed vs. corrected predictions |
| `scenario_1_report.md` | Full results report |

## Key result

Average RMSE across microhabitats reduced from ~5.2 °C (NicheMapR) to ~2.6 °C (RF, 42%)
and ~2.9 °C (LSTM, 39%). RF leads on Sun and Shade; LSTM leads on Air.
To explore how many days of data are needed, run `learning_curve_example.R`.
