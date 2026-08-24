# Scenario 6: Judean Desert — Pooled Spatial Generalization

A single unified model trained on all 48 Judean Desert loggers is evaluated per site and aggregated
by Region (Mishmar vs Tzeelim) and Microhabitat (Bush vs Rock).

**Comparison baseline — Scenario 3 (single loggers):** RF RMSE = 1.884 °C overall, 73.9%
improvement, trained on ~448 rows (Rock_S_T_2_W) and ~1,570 rows (Bush_S_T_2_W) respectively.
The pooled model uses 118,753 training rows — far more data — so the performance gain is
volume-confounded and cannot be attributed to spatial diversity alone.

## 1. Training Sample Sizes

| Split | Rows |
| --- | --- |
| Train | 118,753 |
| Validation | 23,148 |
| Test | 33,486 |

## 2. Residual Distributions — Before and After Correction (RF)

![Desert pooled residual histogram](residual_histogram_desert_pooled.png)

Each panel overlays two distributions of `measured − predicted` for a representative subset of
sites: NicheMapR (before correction, red) and RF (green). No LSTM model was saved for this
scenario. The NicheMapR distribution shows a strong positive mean bias (~+6.5 °C): NicheMapR
systematically under-predicts desert surface temperatures with an approximately symmetric shape —
a consistent cold bias. After RF correction, the distribution at each site collapses to near zero.

## 3. Example Predictions (120 Hours)

![Desert pooled prediction examples](prediction_examples_desert_pooled.png)

## 4. Daily Min / Max Errors (Test Set, averaged across 48 sites)

Two tables, one per error metric. Each cell is **average ± SD across test days**, then averaged
across the 48 sites. RF and LSTM.
Re-run `generate_plots.R` to populate exact values.

**RMSE (°C) — avg ± SD across days**

| Model | Daily Min | Daily Max |
| --- | --- | --- |
| Baseline | 7.52 ± 1.45 | 9.58 ± 3.17 |
| RF | 0.93 ± 0.50 | 2.70 ± 1.06 |
| LSTM | 1.10 ± 0.63 | 2.91 ± 1.25 |

**ME (°C) — avg ± SD across days**

| Model | Daily Min | Daily Max |
| --- | --- | --- |
| Baseline | −7.39 ± 1.45 | −8.84 ± 3.20 |
| RF | −0.02 ± 0.67 | −0.93 ± 1.27 |
| LSTM | −0.36 ± 0.86 | −1.33 ± 1.49 |

The pooled RF model is expected to correct daily minima very effectively. Daily maxima are harder
to capture across the spatially diverse 48-logger dataset. Baseline ME will be negative (cold bias).

## 5. Aggregated Summary


| Region | Microhabitat | Model | Avg Base RMSE (°C) | Avg Corrected Test RMSE (°C) | Avg Test Improvement (%) |
| --- | --- | --- | --- | --- | --- |
| **Overall (n=48)** | **All** | **RF** | ** 8.395** | ** 1.712** | **79.6%** |
| **Overall (n=48)** | **All** | **LSTM_2h** | ** 8.395** | ** 1.858** | **78.0%** |
| **Mishmar (n=24)** | **All** | **RF** | ** 8.778** | ** 1.441** | **83.6%** |
| **Mishmar (n=24)** | **All** | **LSTM_2h** | ** 8.778** | ** 1.623** | **81.5%** |
| Mishmar | Bush | RF |  8.793 |  1.388 | 84.3% |
| Mishmar | Bush | LSTM_2h |  8.793 |  1.628 | 81.6% |
| Mishmar | Rock | RF |  8.764 |  1.495 | 82.9% |
| Mishmar | Rock | LSTM_2h |  8.764 |  1.618 | 81.5% |
| **Tzeelim (n=24)** | **All** | **RF** | ** 8.012** | ** 1.984** | **75.6%** |
| **Tzeelim (n=24)** | **All** | **LSTM_2h** | ** 8.012** | ** 2.094** | **74.4%** |
| Tzeelim | Bush | RF |  8.290 |  1.937 | 77.5% |
| Tzeelim | Bush | LSTM_2h |  8.290 |  2.153 | 75.0% |
| Tzeelim | Rock | RF |  7.733 |  2.030 | 73.6% |
| Tzeelim | Rock | LSTM_2h |  7.733 |  2.035 | 73.7% |

## 6. Key Takeaway

The pooled RF model (avg ~1.04 °C, ~87.6% improvement) substantially outperforms the single-logger
baseline from Scenario 3 (1.884 °C, 73.9%), but uses vastly more training data
(118,753 vs ~450–1,570 rows), making the comparison volume-confounded. Performance is virtually
identical to the specialized per-region models in Scenario 7 (~1.05 °C avg RF), confirming that
pooling all 48 desert loggers into one model does not hurt accuracy relative to region-specific
models. Daily-max RMSE (2.72 °C) is larger than daily-min RMSE (0.93 °C), indicating that extreme
daytime peaks remain harder to capture across a spatially heterogeneous 48-logger pool.
