# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models

Location-specific models are trained on each of the three coastal locations (Ashkelon, Range 24, Rosh HaNikra) and tested on local sensor sites.

## 1. Per-Location Summary
| Location | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- | --- |
| Ashkelon | LSTM_2h | 10.307 | 1.581 | 84.7% |
| Ashkelon | RF | 10.307 | 0.981 | 90.6% |
| Range_24 | LSTM_2h | 7.606 | 1.283 | 83.1% |
| Range_24 | RF | 7.606 | 0.602 | 92.1% |
| Rosh_HaNikra | LSTM_2h | 7.995 | 2.748 | 64.5% |
| Rosh_HaNikra | RF | 7.995 | 0.931 | 88.3% |

## 2. Key Takeaway
Specialized RF models achieve comparable performance to pooled models (see Scenario 4), confirming that the Beach habitat is sufficiently homogeneous for either strategy. The marginal advantage of specialization (~0.02°C) may not justify the cost of maintaining 3 separate models.

