# Scenario 4: Beach Habitat — Pooled Spatial Generalization

A single unified model is trained on all 7 beach logger sites combined and evaluated on each individual site.

**Comparison baseline — Scenario 2 (single logger):** RF RMSE = 3.06 °C, improvement = 62.6%,
trained on ~1,405 rows (Ashkelon 15 m only).
The pooled model uses ~10× more training data, so performance gains cannot be attributed
solely to spatial diversity (see Scenario 8 for a controlled comparison).

## 1. Training Sample Sizes

| Split | Rows |
| --- | --- |
| Train | 13,988 |
| Validation | 4,210 |
| Test | 4,487 |

## 2. Aggregated Summary
| Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- |
| LSTM_2h | 8.544 | 2.092 | 74.5% |
| RF | 8.544 | 0.875 | 89.7% |

## 3. Per-Site Results
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

## 4. Key Takeaway
The pooled RF model (avg 0.875 °C, 89.7% improvement) substantially outperforms the
single-logger baseline from Scenario 2 (3.06 °C, 62.6%), but has ~10× more training data,
so the comparison is not volume-controlled. Performance is virtually identical to the
specialized per-location models in Scenario 5 (~0.84 °C avg), confirming that a single
pooled model generalizes as well as separate location-specific models.
