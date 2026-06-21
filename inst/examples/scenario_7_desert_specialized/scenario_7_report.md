# Scenario 7: Judean Desert — Specialized (Location-Specific) Models

Location-specific models are trained on each of the two desert regions (Mishmar, Tzeelim) and tested on local sensor sites, aggregated by microhabitat type.

## 1. Per-Location & Microhabitat Summary
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

## 2. Key Takeaway
Specialized desert models perform comparably to the pooled model (Scenario 6), with the pooling penalty being effectively zero for RF. This confirms that RF correction is robust to both training strategies in the Judean Desert.

