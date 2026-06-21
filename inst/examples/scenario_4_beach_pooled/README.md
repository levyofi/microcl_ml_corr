# Scenario 4: Beach Habitat — Pooled Spatial Generalization

A single unified model is trained on all 7 beach logger sites combined (13,988 train rows)
and evaluated on each individual site.

**Comparison — Scenario 2 (single logger):** RF RMSE = 3.06 °C, 62.6% improvement, ~1,405 train rows.
The pooled model uses ~10× more data, so gains are volume-confounded (see Scenario 8).

## Run

```r
source(system.file("examples", "scenario_4_beach_pooled", "run_scenario_4.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Beach_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all beach sites |
| `beach_splits.csv` | Pre-defined train/validation/test block indices shared across scenarios 4 and 5 |

## Sample sizes

| Split | Rows |
|-------|------|
| Train | 13,988 |
| Validation | 4,210 |
| Test | 4,487 |

## Outputs

| File | Description |
|------|-------------|
| `results/beach_pooled_results.csv` | Per-site RMSE and improvement % |
| `scenario_4_report.md` | Summary report with aggregated and per-site tables |

## Key result

RF avg RMSE = 0.875 °C (89.7% improvement) — substantially better than the single-logger
baseline in Scenario 2 but with ~10× more training data. Matches the specialized per-location
models in Scenario 5 (~0.84 °C avg RF), confirming that a single pooled model is sufficient.
