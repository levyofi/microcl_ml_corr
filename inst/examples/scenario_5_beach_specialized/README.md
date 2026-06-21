# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models

Separate models are trained per coastal location (Ashkelon, Range 24, Rosh HaNikra)
and tested on the logger sites within each location.
This tests whether location-specific models outperform the single pooled model from Scenario 4.

## Run

```r
source(system.file("examples", "scenario_5_beach_specialized", "run_scenario_5.R", package = "microclCorr"))
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

## Key difference from Scenario 4

Scenario 4 trains one model on all beach sites pooled.
This scenario trains one model per coastal location,
testing whether finer spatial grouping improves correction accuracy.
