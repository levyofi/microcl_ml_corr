# Scenario 7: Judean Desert — Specialized (Location-Specific) Models

Separate models are trained per desert region (Mishmar, Tzeelim)
and tested on the logger sites within each region.
This tests whether region-specific models outperform the single pooled model from Scenario 6.

## Run

```r
source(system.file("examples", "scenario_7_desert_specialized", "run_scenario_7.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `desert_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all 48 desert sites |
| `desert_splits.csv` | Pre-defined train/validation/test block indices shared across scenarios 6 and 7 |

## Outputs

| File | Description |
|------|-------------|
| `results/` | Per-site and aggregated RMSE and R² before and after correction |

## Key difference from Scenario 6

Scenario 6 trains one model on all 48 desert sites pooled.
This scenario trains one model per region,
testing whether finer spatial grouping improves correction accuracy in the desert.
