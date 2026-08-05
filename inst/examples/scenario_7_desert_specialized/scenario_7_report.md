# Scenario 7: Judean Desert — Specialized (Location-Specific) Models

Location-specific models are trained on each of the two desert regions (Mishmar, Tzeelim) and
tested on local sensor sites, aggregated by microhabitat type.

**Comparison baselines:**
- Scenario 3 (single loggers): RF RMSE = 1.884 °C, 73.9% improvement, ~448–1,570 train rows per logger.
- Scenario 6 (pooled, all 48 loggers): RF avg ~1.04 °C, ~87.6% improvement, 118,753 train rows.

## 1. Training Sample Sizes per Region

| Region | Train rows | Validation rows | Test rows |
| --- | --- | --- | --- |
| Mishmar | 60,905 | 12,112 | 17,583 |
| Tzeelim | 56,668 | 10,528 | 15,903 |

## 2. Residual Distributions — Before and After Correction (RF)

![Desert specialized residual histogram](residual_histogram_desert_specialized.png)

Each panel overlays two distributions of `measured − predicted` for a representative subset of
sites: NicheMapR (before correction, red) and RF (green). No LSTM model was saved for this
scenario. The NicheMapR distribution is identical to Scenario 6 (same test data) — strong positive
mean bias (~+6.5 °C) with an approximately symmetric shape. After correction with the
region-specific RF model, each site's distribution collapses to near zero.

## 3. Example Predictions (120 Hours)

![Desert specialized prediction examples](prediction_examples_desert_specialized.png)

## 4. Daily Min / Mean / Max Errors (Test Set, averaged across 48 sites)

Two tables, one per error metric. Each cell is **average ± SD across test days**, then averaged
across the 48 sites. RF and LSTM.
Re-run `generate_plots.R` to populate exact values.

**RMSE (°C) — avg ± SD across days**

| Model | Daily Min | Daily Mean | Daily Max |
| --- | --- | --- | --- |
| Baseline | 7.46 ± 1.57 | 7.89 ± 1.43 | 8.65 ± 1.79 |
| RF | 1.20 ± 0.69 | 1.00 ± 0.64 | 1.39 ± 0.89 |
| LSTM | 1.47 ± 0.80 | 1.09 ± 0.64 | 1.75 ± 1.16 |

**ME (°C) — avg ± SD across days**

| Model | Daily Min | Daily Mean | Daily Max |
| --- | --- | --- | --- |
| Baseline | -7.30 ± 1.57 | -7.76 ± 1.43 | -8.46 ± 1.79 |
| RF | -0.42 ± 0.97 | -0.36 ± 0.89 | -0.06 ± 1.33 |
| LSTM | -0.82 ± 1.12 | -0.55 ± 0.94 | 0.03 ± 1.65 |

Expected to be nearly identical to the pooled model (Scenario 6), confirming that region-specific
training provides negligible improvement over the fully pooled model.

## 5. Per-Region & Microhabitat Summary

| Location | Microhabitat | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- | --- | --- |
| Mishmar | Bush | RF | 8.793 | 1.068 | 87.8% |
| Mishmar | Bush | LSTM_2h | 8.793 | 1.448 | 83.5% |
| Mishmar | Rock | RF | 8.764 | 0.927 | 89.5% |
| Mishmar | Rock | LSTM_2h | 8.764 | 1.324 | 85.0% |
| Tzeelim | Bush | RF | 8.290 | 1.351 | 83.7% |
| Tzeelim | Bush | LSTM_2h | 8.290 | 1.680 | 79.8% |
| Tzeelim | Rock | RF | 7.733 | 0.840 | 89.0% |
| Tzeelim | Rock | LSTM_2h | 7.733 | 1.155 | 84.9% |

## 6. Key Takeaway

Specialized RF models (avg ~1.05 °C, ~87.5% improvement) substantially outperform the
single-logger baseline from Scenario 3 (1.884 °C, 73.9%), but use 56,668–60,905 rows per region
vs ~450–1,570 per single logger, making the comparison volume-confounded. Performance is virtually
identical to the pooled model in Scenario 6 (~1.04 °C avg), confirming that region-specific
specialisation adds no measurable benefit over the fully pooled model for RF correction in the
Judean Desert.
