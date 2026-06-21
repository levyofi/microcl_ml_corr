# Preprocessing Examples

Scripts showing how to prepare raw input files for the `microclCorr` package.
These correspond to the data preparation step shown in Figure 1 of the manuscript.

## Scripts

| Script | Description |
|--------|-------------|
| `preprocessing_single_logger.R` | Prepares data from a single field logger |
| `preprocessing_multiple_loggers.R` | Prepares and pools data from multiple loggers |

Open a script in RStudio and run it with **Source**. Paths are resolved automatically relative to the script location.

## Input data (`data/`)

| File | Description |
|------|-------------|
| `example_logger_single.csv` | Measured temperatures from one field logger (`Rock_S_T_1_W`) |
| `example_nichemapr_single.csv` | NicheMapR predictions for the same site |
| `example_loggers/` | One CSV per logger for three sites (`Rock_S_T_1_W`, `Bush_M_T_1_W`, `Rock_L_M_1_S`) |
| `example_nichemapr_multiple.csv` | NicheMapR predictions for all three sites combined |

All input files are derived from `desert_data_preprocessed.csv` in `inst/extdata/`.

## What the scripts do

Both scripts follow the same steps:

1. **Align** each logger's measured temperatures with its NicheMapR predictions on timestamp
2. **Compute residual** = measured − predicted
3. **Add microhabitat column** if not already present (e.g. `"sun"`, `"shade"`)
4. **Add site ID column** to identify each logger (multiple-logger script only)
5. **Stack** all loggers into one file (multiple-logger script only)

The output file (`data/aligned_single.csv` or `data/aligned_pooled.csv`) is ready to be passed to `load_prepared_csv_data()`.
