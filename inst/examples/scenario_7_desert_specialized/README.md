# Scenario 7: Judean Desert — Specialized (Location-Specific) Models

Separate models are trained per desert region (Mishmar, Tzeelim) and tested on local sites,
with results aggregated by microhabitat type.

**Comparisons:**
- Scenario 3 (single loggers): RF RMSE = 1.884 °C, 73.9%, ~448–1,570 train rows per logger.
- Scenario 6 (pooled, all 48 loggers): RF avg ~1.04 °C, ~87.6%, 118,753 train rows.

## Run

```r
source(system.file("examples", "scenario_7_desert_specialized", "run_scenario_7.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `desert_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all 48 desert sites |
| `desert_splits.csv` | Pre-defined train/validation/test block indices shared across scenarios 6 and 7 |

## Sample sizes per region

| Region | Train rows | Validation rows |
|--------|------------|-----------------|
| Mishmar | 81,428 | 16,620 |
| Tzeelim | 56,668 | 10,528 |

## Outputs

| File | Description |
|------|-------------|
| `results/desert_specialized_results.csv` | Per-site RMSE and improvement % |
| `scenario_7_report.md` | Summary report per region and microhabitat |

## Key result

RF avg ~1.05 °C (~87.5% improvement) — matches the fully pooled model (Scenario 6, ~1.04 °C),
confirming that region-specific specialisation adds no benefit in the desert. Both substantially
outperform the single-logger baseline (Scenario 3, 1.884 °C) but use far more training data.
