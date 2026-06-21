# Scenario 5: Beach Habitat — Specialized (Location-Specific) Models

Location-specific models are trained on each of the three coastal locations (Ashkelon, Range 24, Rosh HaNikra) and tested on local sensor sites.

**Comparison baselines:**
- Scenario 2 (single logger, Ashkelon 15 m only): RF RMSE = 3.06 °C, 62.6% improvement, ~1,405 train rows.
- Scenario 4 (pooled, all 7 loggers): RF avg RMSE = 0.875 °C, 89.7% improvement, 13,988 train rows.

## 1. Training Sample Sizes per Location

| Location | Train rows | Validation rows |
| --- | --- | --- |
| Ashkelon | 4,631 | 1,343 |
| Range_24 | 4,464 | 1,308 |
| Rosh_HaNikra | 4,893 | 1,559 |

## 2. Per-Location Summary
| Location | Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- | --- |
| Ashkelon | LSTM_2h | 10.307 | 1.581 | 84.7% |
| Ashkelon | RF | 10.307 | 0.981 | 90.6% |
| Range_24 | LSTM_2h | 7.606 | 1.283 | 83.1% |
| Range_24 | RF | 7.606 | 0.602 | 92.1% |
| Rosh_HaNikra | LSTM_2h | 7.995 | 2.748 | 64.5% |
| Rosh_HaNikra | RF | 7.995 | 0.931 | 88.3% |

## 3. Key Takeaway
Specialized RF models (avg ~0.84 °C, ~90% improvement) substantially outperform the
single-logger baseline from Scenario 2 (3.06 °C, 62.6%), but use ~3× more training data
per location (4,464–4,893 rows vs ~1,405 rows), making it a volume-confounded comparison.
Performance is virtually identical to the pooled model in Scenario 4 (0.875 °C), which uses
all 13,988 rows — confirming that location-specific pooling captures most of the benefit of
the full pooled model while using only a third of the data.
