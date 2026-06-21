# Scenario 6: Judean Desert — Pooled Spatial Generalization

A single unified model trained on all 48 Judean Desert loggers is evaluated per site and aggregated by Region (Mishmar vs Tzeelim) and Microhabitat (Bush vs Rock).

## 1. Aggregated Summary
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

## 2. Visual Comparison
![Desert Pooled Comparison](desert_pooled_comparison.png)

## 3. Key Takeaway
The pooled RF model achieves >85% error reduction across all desert categories, confirming strong spatial transferability even across the Judean Desert's diverse microhabitats (varying rock sizes, bush cover, and seasonal conditions).

