# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models

Separate models are trained per coastal location (Ashkelon, Range 24, Rosh HaNikra)
and tested on the logger sites within each location.

**Comparisons:**
- Scenario 2 (single logger, Ashkelon 15 m): RF RMSE = 3.06 °C, 62.6%, ~1,405 train rows.
- Scenario 4 (pooled, all 7 loggers): RF avg RMSE = 0.875 °C, 89.7%, 13,988 train rows.

## Run

```r
source(system.file("examples", "scenario_5_beach_specialized", "run_scenario_5.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Beach_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all beach sites |
| `beach_splits.csv` | Pre-defined train/validation/test block indices shared across scenarios 4 and 5 |

## Sample sizes per location

| Location | Train rows | Validation rows |
|----------|------------|-----------------|
| Ashkelon | 4,631 | 1,343 |
| Range_24 | 4,464 | 1,308 |
| Rosh_HaNikra | 4,893 | 1,559 |

## Outputs

| File | Description |
|------|-------------|
| `results/beach_specialized_results.csv` | Per-site RMSE and improvement % |
| `scenario_5_report.md` | Summary report with per-location tables |

## Key result

RF avg ~0.84 °C (~90% improvement) — matches the fully pooled model (Scenario 4, 0.875 °C)
while using only ~1/3 of the training data. Substantially outperforms the single-logger
baseline (Scenario 2, 3.06 °C) but uses ~3× more data per location.
