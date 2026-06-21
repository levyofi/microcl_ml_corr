# Scenario 7: Judean Desert — Specialized (Location-Specific) Models

Location-specific models are trained on each of the two desert regions (Mishmar, Tzeelim) and tested on local sensor sites, aggregated by microhabitat type.

**Comparison baselines:**
- Scenario 3 (single loggers): RF RMSE = 1.884 °C, 73.9% improvement, ~448–1,570 train rows per logger.
- Scenario 6 (pooled, all 48 loggers): RF avg ~1.04 °C, ~87.6% improvement, 118,753 train rows.

## 1. Training Sample Sizes per Region

| Region | Train rows | Validation rows |
| --- | --- | --- |
| Mishmar | 81,428 | 16,620 |
| Tzeelim | 56,668 | 10,528 |

## 2. Per-Location & Microhabitat Summary
| Location | Microhabitat | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- | --- | --- |
| Mishmar | Bush | LSTM_2h | 8.793 | 1.448 | 83.5% |
| Mishmar | Bush | RF | 8.793 | 1.068 | 87.8% |
| Mishmar | Rock | LSTM_2h | 8.764 | 1.324 | 85.0% |
| Mishmar | Rock | RF | 8.764 | 0.927 | 89.5% |
| Tzeelim | Bush | LSTM_2h | 8.290 | 1.680 | 79.8% |
| Tzeelim | Bush | RF | 8.290 | 1.351 | 83.7% |
| Tzeelim | Rock | LSTM_2h | 7.733 | 1.155 | 84.9% |
| Tzeelim | Rock | RF | 7.733 | 0.840 | 89.0% |

## 3. Key Takeaway
Specialized RF models (avg ~1.05 °C, ~87.5% improvement) substantially outperform the
single-logger baseline from Scenario 3 (1.884 °C, 73.9%), but use 56,668–81,428 rows
vs ~450–1,570 per single logger, making the comparison volume-confounded.
Performance is virtually identical to the pooled model in Scenario 6 (~1.04 °C avg),
confirming that region-specific specialisation adds no measurable benefit over the
fully pooled model for Random Forest correction in the Judean Desert.
