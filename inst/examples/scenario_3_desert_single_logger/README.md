# Scenario 3: Judean Desert Habitat (Tzeelim)

Local microclimate correction for loggers placed under a desert bush and on a desert rock
at the Tzeelim site in the Judean Desert.
Desert environments have high daily meteorological consistency,
making them the easiest habitat for the correction models.

## Run

```r
source(system.file("examples", "scenario_3_desert_single_logger", "run_scenario_3.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `desert_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for the Tzeelim desert site |

## Outputs

| File | Description |
|------|-------------|
| `results/` | CSV with RMSE and R² before and after correction |
| `prediction_examples_desert.png` | 120-hour observed vs. corrected predictions |
| `learning_curves_desert.png` | RMSE as a function of training data size |
| `scenario_3_report.md` | Full results report |

## Key result

RMSE reduced from ~7.2 °C (NicheMapR baseline) to ~1.9 °C (RF) and ~2.3 °C (LSTM),
an improvement of up to 74%. Even a single day of training data captures >90% of the
maximum improvement, reflecting the high daily consistency of desert conditions.
