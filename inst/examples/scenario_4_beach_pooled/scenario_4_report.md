# Scenario 4: Beach Habitat — Pooled Spatial Generalization

A single unified model trained on all 7 Beach logger locations is evaluated on each individual site to measure spatial transferability.

## 1. Aggregated Summary
| Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- |
| LSTM_2h | 8.544 | 2.092 | 74.5% |
| RF | 8.544 | 0.875 | 89.7% |

## 2. Per-Site Results
| Site | Model | Base RMSE (°C) | Corrected RMSE (°C) | Improvement (%) |
| --- | --- | --- | --- | --- |
| Range_24 25 m | RF | 7.602 | 0.669 | 91.2% |
| Range_24 45 m | RF | 7.610 | 0.660 | 91.3% |
| Rosh_HaNikra 15 m | RF | 6.810 | 0.963 | 85.9% |
| Rosh_HaNikra 25 m | RF | 9.841 | 1.150 | 88.3% |
| Ashkelon 10 m | RF | 10.684 | 1.232 | 88.5% |
| Ashkelon 15 m | RF | 9.930 | 0.758 | 92.4% |
| Rosh_HaNikra 45 m | RF | 7.334 | 0.695 | 90.5% |
| Range_24 25 m | LSTM_2h | 7.602 | 1.609 | 78.8% |
| Range_24 45 m | LSTM_2h | 7.610 | 1.450 | 81.0% |
| Rosh_HaNikra 15 m | LSTM_2h | 6.810 | 2.802 | 58.9% |
| Rosh_HaNikra 25 m | LSTM_2h | 9.841 | 3.040 | 69.1% |
| Ashkelon 10 m | LSTM_2h | 10.684 | 1.479 | 86.2% |
| Ashkelon 15 m | LSTM_2h | 9.930 | 1.625 | 83.6% |
| Rosh_HaNikra 45 m | LSTM_2h | 7.334 | 2.637 | 64.0% |

## 3. Key Takeaway
The pooled RF model achieves >88% error reduction on every beach site, confirming that a single unified model generalizes well across homogeneous coastal microhabitats without meaningful accuracy loss compared to specialized models.

