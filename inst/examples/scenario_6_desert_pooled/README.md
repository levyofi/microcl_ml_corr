# Scenario 6: Judean Desert — Pooled Spatial Generalization

A single unified model is trained on all 48 desert logger sites combined
and tested on each individual site.
Results are aggregated by region (Mishmar, Tzeelim) and microhabitat (rock, bush).

## Run

```r
source(system.file("examples", "scenario_6_desert_pooled", "run_scenario_6.R", package = "microclCorr"))
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

## Key difference from Scenario 3

Scenario 3 trains a dedicated model on a single desert site.
This scenario trains one model on all 48 desert sites pooled together,
testing large-scale spatial generalisation across the Judean Desert.
