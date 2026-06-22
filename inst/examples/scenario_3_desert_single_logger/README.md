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
| `results/` | Per-microhabitat CSV with RMSE before and after correction |
| `prediction_examples_desert.png` | 120-hour observed vs. corrected predictions |
| `scenario_3_report.md` | Full results report |

## Key result

Average RMSE reduced from ~6.4 °C (NicheMapR baseline) to ~1.9 °C for both models (~71% improvement).
RF leads on Rock; LSTM leads on Bush. To find the minimum training days needed,
run `learning_curve_example.R`.
