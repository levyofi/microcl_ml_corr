# Scenario 8: Zero-Shot Spatial Transfer

A correction model is trained exclusively on data from neighbouring sites and applied to
a new location where no logger data exists. Includes a downsampled pooled control to
separate the effect of training data volume from spatial diversity.

**Comparison — Scenario 2 (single logger, Ashkelon 15 m):** RF RMSE = 3.06 °C, 62.6%,
~1,405 train rows. Note: the "Specialized (Local Data)" condition here uses all Ashkelon
loggers (~4,631 rows), not just one, so it already has ~3× more data than Scenario 2.

## Run

```r
source(system.file("examples", "scenario_8_zero_shot_transfer", "run_scenario_8.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `Beach_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all beach sites |
| `beach_splits.csv` | Pre-defined train/validation/test block indices |

## Sample sizes per strategy (Ashkelon as target example)

| Strategy | Train rows |
|----------|------------|
| Zero-Shot (Nearby Sites) | 9,357 |
| Specialized (Local Data) | 4,631 |
| Pooled (All Sites) | 13,988 |
| Pooled (Downsampled to N) | 4,631 |

## Outputs

| File | Description |
|------|-------------|
| `results/zero_shot_results.csv` | RMSE and improvement % per target location and strategy |
| `scenario_8_report.md` | Full results with key findings |

## Key result

Zero-shot transfer achieves 58–68% improvement with no local data at all. The downsampled
pooled control (same N as local, drawn from other sites) yields 73–84% improvement vs.
89–92% for local data — confirming that local data is more informative per row than
data from other sites, even when volume is equalised.
