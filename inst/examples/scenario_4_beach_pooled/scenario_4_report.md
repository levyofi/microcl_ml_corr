# Scenario 4: Beach Habitat — Pooled Spatial Generalization

A single unified model is trained on all 7 beach logger sites combined and evaluated on each
individual site.

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

## 2. Residual Distributions — Before and After Correction (RF)

![Beach pooled residual histogram](residual_histogram_beach_pooled.png)

Each panel overlays two distributions of `measured − predicted` per site: NicheMapR (before
correction, red) and RF (green), and LSTM (blue). The NicheMapR
distribution across all 7 beach loggers is strongly left-skewed (mean ≈ −3.7 °C): NicheMapR
over-predicts coastal surface temperatures, with extreme daytime over-predictions. After RF
correction the distribution collapses to near zero at every site.

## 3. Example Predictions (120 Hours)

![Beach pooled prediction examples](prediction_examples_beach_pooled.png)

## 4. Daily Min / Mean / Max Errors (Test Set, averaged across 7 sites)

Two tables, one per error metric. Each cell is **average ± SD across test days**, then averaged
across the 7 sites.

**RMSE (°C) — avg ± SD across days**

| Model | Daily Min | Daily Mean | Daily Max |
| --- | --- | --- | --- |
| Baseline | 3.04 ± 1.67 | 4.24 ± 1.76 | 15.04 ± 5.79 |
| RF | 0.50 ± 0.34 | 0.31 ± 0.20 | 0.90 ± 0.60 |
| LSTM | 4.32 ± 2.57 | 3.15 ± 1.49 | 5.22 ± 2.93 |

**ME (°C) — avg ± SD across days** (positive = over-prediction, negative = under-prediction)

| Model | Daily Min | Daily Mean | Daily Max |
| --- | --- | --- | --- |
| Baseline | −2.44 ± 1.81 | +1.45 ± 3.88 | +12.38 ± 8.26 |
| RF | −0.01 ± 0.46 | −0.06 ± 0.29 | −0.02 ± 0.79 |
| LSTM | −1.87 ± 3.74 | −0.87 ± 2.68 | −0.66 ± 4.56 |

The pooled RF model achieves outstanding daily-max correction (0.90 vs 15.04 °C baseline, 94%
reduction) with near-zero ME, far better than the single-logger model in Scenario 2.
LSTM performs substantially worse than RF here, suggesting the pooled LSTM did not generalise
as well across the 7 beach sites.

## 5. Aggregated Hourly Summary

| Model | Avg Base RMSE (°C) | Avg Corrected RMSE (°C) | Avg Improvement (%) |
| --- | --- | --- | --- |
| RF | 8.544 | 0.875 | 89.7% |
| LSTM_2h | 8.544 | 2.092 | 74.5% |

## 6. Per-Site Results

| Site | Model | Base RMSE (°C) | Corrected RMSE (°C) | Improvement (%) |
| --- | --- | --- | --- | --- |
| Ashkelon 10 m | RF | 10.684 | 1.232 | 88.5% |
| Ashkelon 10 m | LSTM_2h | 10.684 | 1.479 | 86.2% |
| Ashkelon 15 m | RF | 9.930 | 0.758 | 92.4% |
| Ashkelon 15 m | LSTM_2h | 9.930 | 1.625 | 83.6% |
| Range_24 25 m | RF | 7.602 | 0.669 | 91.2% |
| Range_24 25 m | LSTM_2h | 7.602 | 1.609 | 78.8% |
| Range_24 45 m | RF | 7.610 | 0.660 | 91.3% |
| Range_24 45 m | LSTM_2h | 7.610 | 1.450 | 81.0% |
| Rosh_HaNikra 15 m | RF | 6.810 | 0.963 | 85.9% |
| Rosh_HaNikra 15 m | LSTM_2h | 6.810 | 2.802 | 58.9% |
| Rosh_HaNikra 25 m | RF | 9.841 | 1.150 | 88.3% |
| Rosh_HaNikra 25 m | LSTM_2h | 9.841 | 3.040 | 69.1% |
| Rosh_HaNikra 45 m | RF | 7.334 | 0.695 | 90.5% |
| Rosh_HaNikra 45 m | LSTM_2h | 7.334 | 2.637 | 64.0% |

## 7. Key Takeaway

The pooled RF model (avg 0.875 °C, 89.7% improvement) substantially outperforms the single-logger
baseline from Scenario 2 (3.06 °C, 62.6%), but has ~10× more training data, so the comparison is
not volume-controlled. Performance is virtually identical to the specialized per-location models
in Scenario 5 (~0.84 °C avg), confirming that a single pooled model generalizes as well as
separate location-specific models. Daily-max RMSE (0.90 °C pooled vs 1.39 °C single-logger)
shows the pooled model's particular strength at capturing extreme temperature events.
