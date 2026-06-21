# Scenario 8: Zero-Shot Spatial Transfer

A correction model is trained exclusively on data from neighbouring sites
and applied to a new location where no logger data exists.
This is the practical deployment scenario: correcting NicheMapR predictions
at a site before any field measurements have been collected.

Also includes a scientific control that separates the effect of training data
volume from spatial diversity, to confirm that performance gains come from
seeing diverse locations rather than simply from more data.

## Run

```r
source(system.file("examples", "scenario_8_zero_shot_transfer", "run_scenario_8.R", package = "microclCorr"))
```

## Input

| File | Description |
|------|-------------|
| `desert_data_preprocessed.csv` | Pre-aligned logger + NicheMapR data for all 48 desert sites |
| `desert_splits.csv` | Pre-defined train/validation/test block indices |

## Outputs

| File | Description |
|------|-------------|
| `results/` | RMSE and R² for zero-shot transfer vs. local and pooled baselines |

## Key difference from Scenarios 6 and 7

Scenarios 6 and 7 always include some data from the test site during training.
This scenario deliberately withholds all data from the target site,
simulating a true zero-shot transfer to an unseen location.
