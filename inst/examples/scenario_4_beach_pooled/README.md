# Scenario 4: Beach Habitat — Pooled Spatial Generalization

A single unified model is trained on data from all beach logger sites combined
and then tested on each individual site.
This tests whether one shared model can generalise across different coastal locations.

## Run

```r
source(system.file("examples", "scenario_4_beach_pooled", "run_scenario_4.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Beach_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all beach sites |
| `beach_splits.csv` | Pre-defined train/validation/test block indices shared across scenarios 4 and 5 |

## Outputs

| File | Description |
|------|-------------|
| `results/` | Per-site RMSE and R² before and after correction |

## Key difference from Scenario 2

Scenario 2 trains a dedicated model per site (local correction).
This scenario trains one model on all sites pooled together and evaluates
how well it transfers to each individual location.
