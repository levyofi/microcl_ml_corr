# microclCorr — Examples

This folder contains ready-to-run R scripts that demonstrate every step of the
`microclCorr` pipeline, from raw CSV files to corrected microclimate predictions.

---

## Before you start — what does the pipeline do?

NicheMapR is a physical model that predicts local microclimate temperatures.
It is not perfect: there is always a gap between its prediction and what a
temperature logger actually measured. This gap is called the **residual**:

```
residual = measured temperature − NicheMapR prediction
```

`microclCorr` trains a machine learning model to predict that residual.
The corrected temperature is then:

```
corrected temperature = NicheMapR prediction + predicted residual
```

Two model types are compared in every scenario:

| Model | Description |
|-------|-------------|
| **Random Forest (RF)** | An ensemble of decision trees. Fast, robust, and works without data normalisation. |
| **LSTM** | A neural network designed for time-series data. Looks at the past 2 hours of measurements to predict the current residual. |

Accuracy is measured by **RMSE** (Root Mean Squared Error, in °C). Lower RMSE = smaller average error. **Improvement %** = how much the model reduced the original NicheMapR error.

---

## Folder structure

```
examples/
├── README.md                      ← this file
├── utils.R                        ← shared helper functions (loaded automatically)
│
├── preprocessing_examples/        ← Step 0: prepare your own CSV files
│   ├── README.md
│   ├── preprocessing_single_logger.R
│   ├── preprocessing_multiple_loggers.R
│   └── data/                      ← example input CSVs
│
├── scenario_1_valley_single_logger/             ← Single logger, Mediterranean valley
├── scenario_2_beach_single_logger/              ← Single logger, coastal beach
├── scenario_3_desert_single_logger/             ← Single logger, Judean desert
├── scenario_4_beach_pooled/       ← All beach loggers pooled into one model
├── scenario_5_beach_specialized/  ← One model per coastal location
├── scenario_6_desert_pooled/      ← All 48 desert loggers pooled
├── scenario_7_desert_specialized/ ← One model per desert region
└── scenario_8_zero_shot_transfer/ ← Apply a model to a site with no local data
```

Each scenario folder contains:
- `run_scenario_N.R` — the script to run
- `results/` — CSV files with RMSE results
- `scenario_N_report.md` — summary report with tables
- `README.md` — description, sample sizes, and comparison with related scenarios

---

## How to run a scenario

Open any `run_scenario_N.R` file in RStudio and click **Source**, or run from
the repository root:

```r
source("inst/examples/scenario_1_valley_single_logger/run_scenario_1.R")
```

All input data is bundled with the package and located automatically via
`system.file()`. Results are written to the `results/` subfolder of each scenario.

---

## Which scenario should I start with?

| Scenario | What it answers |
|----------|----------------|
| **preprocessing_examples** | How do I prepare my own logger CSV before using this package? |
| **1 — Valley** | How well does local correction work? How much data do I need? |
| **2 — Beach** | Same as Scenario 1, but for a coastal site (harder to correct). |
| **3 — Desert** | Same as Scenario 1, but for a desert site (very easy to correct). |
| **4 — Beach Pooled** | Does pooling all loggers into one model improve things? |
| **5 — Beach Specialized** | Does training one model per location beat the pooled model? |
| **6 — Desert Pooled** | Same as Scenario 4, but for 48 desert loggers. |
| **7 — Desert Specialized** | Same as Scenario 5, but per desert region. |
| **8 — Zero-Shot Transfer** | Can I correct a new site where no logger data exists? |

**Recommended reading order:** preprocessing → 1 → 4 → 8.
Scenarios 2, 3, 5, 6, 7 follow the same pattern as 1 and 4.

---

## Key findings across scenarios

All values at 42 days of training data where applicable. RMSE in °C; lower is better.

| Scenario | RF RMSE | RF improvement | LSTM RMSE | LSTM improvement |
|----------|---------|---------------|-----------|-----------------|
| 1 — Valley (single logger, avg across microhabitats) | 2.59 | 42% | 2.88 | 39% |
| 2 — Beach (single logger) | 1.40 | 88% | 2.50 | 79% |
| 3 — Desert (single logger, avg across microhabitats) | 1.87 | 71% | 1.89 | 71% |
| 4 — Beach Pooled (13,988 train rows) | 0.88 | 90% | 2.09 | 75% |
| 5 — Beach Specialized (4,631–4,893 rows/location) | ~0.84 | ~90% | ~1.87 | ~77% |
| 6 — Desert Pooled (118,753 train rows) | ~1.04 | ~88% | ~1.51 | ~83% |
| 7 — Desert Specialized (57k–81k rows/region) | ~1.05 | ~88% | ~1.40 | ~83% |
| 8 — Zero-Shot (no local data) | ~3.0 | ~64% | — | — (RF only; see note) |

RF consistently outperforms LSTM on single-logger scenarios; the gap narrows
with larger pooled datasets. Pooled and specialized models outperform
single-logger models, but they also use much more training data.
Scenario 8 shows that even without any local data, a model trained on nearby
sites still reduces NicheMapR error by ~64%.

> **Note — Scenario 8 (Zero-Shot):** only Random Forest is tested. The
> zero-shot experiment focuses on whether *spatial generalisation* works at
> all, which RF answers cleanly. Adding LSTM would require windowing and
> scaling across four training strategies × 10 downsampling runs, with
> results expected to mirror the RF findings.
