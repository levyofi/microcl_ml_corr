# Scenario 6: Judean Desert — Pooled Spatial Generalization

A single unified model is trained on all 48 desert logger sites combined (118,753 train rows)
and tested on each individual site, with results aggregated by region and microhabitat.

**Comparison — Scenario 3 (single loggers):** RF RMSE = 1.884 °C, 73.9% improvement,
~448–1,570 train rows per logger. The pooled model uses far more data, so gains are
volume-confounded.

## Run

```r
source(system.file("examples", "scenario_6_desert_pooled", "run_scenario_6.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `desert_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all 48 desert sites |
| `desert_splits.csv` | Pre-defined train/validation/test block indices shared across scenarios 6 and 7 |

## Sample sizes

| Split | Rows |
|-------|------|
| Train | 118,753 |
| Validation | 23,148 |
| Test | 33,486 |

## Outputs

| File | Description |
|------|-------------|
| `results/desert_pooled_results.csv` | Per-site RMSE and improvement % |
| `scenario_6_report.md` | Summary report aggregated by region and microhabitat |

## Key result

RF avg ~1.04 °C (~87.6% improvement) — substantially better than the single-logger baseline
in Scenario 3 (1.884 °C, 73.9%) but with vastly more training data. Matches the specialized
per-region models in Scenario 7 (~1.05 °C), confirming that one pooled model suffices across
the full Judean Desert.
