# Scenario 8: Zero-Shot Spatial Transfer — Training on Nearby Sites

## Background
In practice, users often want to correct NicheMapR predictions at a **new location where no temperature logger has been deployed**. This scenario evaluates whether a Random Forest model trained on data from neighboring sites can provide meaningful correction at an unseen target site.

**Comparison baseline — Scenario 2 (single logger, Ashkelon 15 m only):** RF RMSE = 3.06 °C,
62.6% improvement, ~1,405 train rows. The "Specialized (Local Data)" condition below uses
all Ashkelon loggers combined (~4,631 rows), so it already has ~3× more data than the
Scenario 2 single-logger baseline.

## 1. Logger Temperature Statistics (Full Dataset)

Daily mean, daily minimum, and daily maximum temperatures averaged across all days (mean ± SD
across days), for all sites within each beach location.

| Location | Daily mean (°C) | Daily min (°C) | Daily max (°C) |
| --- | --- | --- | --- | --- | --- | --- |
| Ashkelon     | 29.92 ± 4.17 | 23.37 ± 2.99 | 39.38 ± 7.39 |
| Range_24     | 32.12 ± 2.02 | 23.04 ± 2.40 | 46.73 ± 3.09 |
| Rosh_HaNikra | 32.66 ± 1.87 | 23.88 ± 3.28 | 7958 | 45.85 ± 5.37 |

## 2. Residual Distributions — Before and After Correction (Zero-Shot RF)

![Zero-shot residual histogram](residual_histogram_zero_shot.png)

Each column shows one held-out target location. NicheMapR (red, before) is strongly left-skewed
with extreme negative residuals reaching −30 °C — the large daytime over-prediction signal.
The zero-shot RF (green, after) collapses to near zero but with noticeably wider spread than the
specialized/pooled models in Scenarios 4–5, consistent with the higher zero-shot RMSE (~2.5–3.6 °C
vs ~0.6–1.1 °C for local models).

## 3. Example Predictions — Zero-Shot Strategy (120 Hours)

![Zero-shot prediction examples](prediction_examples_zero_shot.png)

## 4. Daily Min / Mean / Max Errors — Zero-Shot Strategy (Test Set)

Two tables for strategy A (zero-shot, trained on other 2 locations). Each cell is **average ± SD
across test days**. No LSTM in this scenario (RF only).

**RMSE (°C) — avg ± SD across days**

| Location | Model | Daily Min | Daily Mean | Daily Max |
| --- | --- | --- | --- | --- | --- | --- |
| Ashkelon | Baseline | 2.83 ± 1.34 | 7479 | 4.49 ± 1.81 | 20.09 ± 8.45 |
| Ashkelon | RF (zero-shot) | 1.57 ± 0.99 | 7479 | 1.68 ± 0.86 | 5.98 ± 3.18 |
| Range_24 | Baseline | 2.20 ± 1.34 | 7248 | 4.03 ± 1.58 | 12.75 ± 4.69 |
| Range_24 | RF (zero-shot) | 1.61 ± 0.95 | 7248 | 0.90 ± 0.51 | 2.92 ± 1.49 |
| Rosh_HaNikra | Baseline | 3.81 ± 1.95 | 7958 | 3.92 ± 1.73 | 16.09 ± 6.92 |
| Rosh_HaNikra | RF (zero-shot) | 2.29 ± 0.97 | 7958 | 1.39 ± 0.88 | 4.69 ± 2.49 |

**ME (°C) — avg ± SD across days** (positive = over-prediction, negative = under-prediction)

| Location | Model | Daily Min | Daily Mean | Daily Max |
| --- | --- | --- | --- | --- | --- | --- |
| Ashkelon | Baseline | −2.47 ± 1.40 | 7479 | +2.94 ± 3.43 | +18.19 ± 8.63 |
| Ashkelon | RF (zero-shot) | −0.46 ± 1.52 | 7479 | +1.18 ± 1.22 | +4.84 ± 3.55 |
| Range_24 | Baseline | −1.61 ± 1.53 | 7248 | +1.25 ± 3.90 | +8.68 ± 9.49 |
| Range_24 | RF (zero-shot) | +0.94 ± 1.32 | 7248 | +0.53 ± 0.74 | −1.73 ± 2.39 |
| Rosh_HaNikra | Baseline | −3.08 ± 2.27 | 7958 | +0.98 ± 3.84 | +12.83 ± 9.83 |
| Rosh_HaNikra | RF (zero-shot) | −0.88 ± 2.14 | 7958 | −0.90 ± 1.07 | +1.42 ± 4.53 |

The zero-shot model substantially reduces daily-max RMSE (e.g. Ashkelon: 20.09 → 5.98 °C) but
leaves residual errors much larger than the specialized/pooled models (~1.4 °C daily-max).
ME after zero-shot correction is near-zero for daily means but shows larger day-to-day variability
(SD ~1–1.2 °C) compared to locally trained models.

## 5. Experimental Design
For each Beach location, we:
1. **Zero-Shot (Nearby Sites)**: Train RF on data from the other 2 locations only, excluding all target-site data entirely. This simulates deploying a correction model to a new field site.
2. **Specialized (Local Data)**: Train RF on local data only (upper bound for comparison).
3. **Pooled (All Sites)**: Train on all 3 locations including the target (best case).
4. **Pooled (Downsampled to N)**: Train on a random sample of the pooled data matching the local dataset size. This controls for the effect of training set volume.

## 6. Results
| Target Location | Training Strategy | Train Size | Test Size | Corrected RMSE (°C) | Raw NicheMapR (°C) | Improvement (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Ashkelon | Zero-Shot (Nearby Sites) | 9357 | 7479 | 3.284 | 10.403 | 68.4% |
| Ashkelon | Specialized (Local Data) | 4631 | 7479 | 1.089 | 10.403 | 89.5% |
| Ashkelon | Pooled (All Sites) | 13988 | 7479 | 1.088 | 10.403 | 89.5% |
| Ashkelon | Pooled (Downsampled to N) | 4631 | 7479 | 1.721 | 10.403 | 83.5% |
| Range_24 | Zero-Shot (Nearby Sites) | 9524 | 7248 | 2.488 | 7.585 | 67.2% |
| Range_24 | Specialized (Local Data) | 4464 | 7248 | 0.605 | 7.585 | 92.0% |
| Range_24 | Pooled (All Sites) | 13988 | 7248 | 0.664 | 7.585 | 91.2% |
| Range_24 | Pooled (Downsampled to N) | 4464 | 7248 | 1.459 | 7.585 | 80.8% |
| Rosh_HaNikra | Zero-Shot (Nearby Sites) | 9095 | 7958 | 3.615 | 8.552 | 57.7% |
| Rosh_HaNikra | Specialized (Local Data) | 4893 | 7958 | 1.017 | 8.552 | 88.1% |
| Rosh_HaNikra | Pooled (All Sites) | 13988 | 7958 | 1.035 | 8.552 | 87.9% |
| Rosh_HaNikra | Pooled (Downsampled to N) | 4893 | 7958 | 2.303 | 8.552 | 73.1% |

## 7. Visual Summary — All Strategies
![Zero-Shot Transfer Comparison](zero_shot_transfer.png)

## 8. Key Findings

### Zero-Shot Transfer Provides Substantial Correction
Even without any local training data, the zero-shot model reduces NicheMapR error by **58-68%** across all Beach locations. This confirms that the physical feature representation (radiation, humidity, wind speed, temporal encoding) captures generalizable correction patterns that transfer across sites.

### The Gap to Local Models
The zero-shot corrected RMSE (~2.5-3.6°C) is notably higher than locally-trained models (~0.6-1.1°C), indicating that **site-specific physical parameters** (localized albedo, wind blocks, terrain shading) cannot be fully resolved without some local data representation.

### Practical Recommendation
For a new field site where no logger data is available, deploying a zero-shot correction model trained on nearby regional loggers provides a meaningful first-pass correction (**>58% error reduction**) over raw NicheMapR output. Once even a small amount of local logger data becomes available, retraining as a specialized or pooled model will dramatically improve accuracy.

