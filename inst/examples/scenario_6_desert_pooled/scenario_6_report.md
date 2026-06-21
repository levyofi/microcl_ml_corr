# Scenario 6: Judean Desert — Pooled Spatial Generalization

A single unified model trained on all 48 Judean Desert loggers is evaluated per site and aggregated by Region (Mishmar vs Tzeelim) and Microhabitat (Bush vs Rock).

**Comparison baseline — Scenario 3 (single loggers):** RF RMSE = 1.884 °C overall, 73.9% improvement,
trained on ~448 rows (Rock_S_T_2_W) and ~1,570 rows (Bush_S_T_2_W) respectively.
The pooled model uses 118,753 training rows — far more data — so the performance gain
is volume-confounded and cannot be attributed to spatial diversity alone.

## 1. Training Sample Sizes

| Split | Rows |
| --- | --- |
| Train | 118,753 |
| Validation | 23,148 |
| Test | 33,486 |

## 2. Aggregated Summary
| Region | Microhabitat | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- | --- | --- |
| Mishmar | Bush | LSTM_2h | 8.793 | 1.385 | 84.2% |
| Mishmar | Rock | LSTM_2h | 8.764 | 1.345 | 84.7% |
| Tzeelim | Bush | LSTM_2h | 8.290 | 1.709 | 79.4% |
| Tzeelim | Rock | LSTM_2h | 7.733 | 1.185 | 84.5% |
| Mishmar | Bush | RF | 8.793 | 1.067 | 87.8% |
| Mishmar | Rock | RF | 8.764 | 0.920 | 89.6% |
| Tzeelim | Bush | RF | 8.290 | 1.347 | 83.8% |
| Tzeelim | Rock | RF | 7.733 | 0.826 | 89.2% |

## 3. Key Takeaway
The pooled RF model (avg ~1.04 °C, ~87.6% improvement) substantially outperforms the
single-logger baseline from Scenario 3 (1.884 °C, 73.9%), but uses vastly more training data
(118,753 vs ~450–1,570 rows), making the comparison volume-confounded.
Performance is virtually identical to the specialized per-region models in Scenario 7
(~1.05 °C avg RF), confirming that pooling all 48 desert loggers into one model
does not hurt accuracy relative to region-specific models.
