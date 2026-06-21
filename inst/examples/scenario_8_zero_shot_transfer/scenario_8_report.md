# Scenario 8: Zero-Shot Spatial Transfer — Training on Nearby Sites

## Motivation
In practice, users often want to correct NicheMapR predictions at a **new location where no temperature logger has been deployed**. This scenario evaluates whether a Random Forest model trained on data from neighboring sites can provide meaningful correction at an unseen target site.

**Comparison baseline — Scenario 2 (single logger, Ashkelon 15 m only):** RF RMSE = 3.06 °C,
62.6% improvement, ~1,405 train rows. The "Specialized (Local Data)" condition below uses
all Ashkelon loggers combined (~4,631 rows), so it already has ~3× more data than the
Scenario 2 single-logger baseline.

## Experimental Design
For each Beach location, we:
1. **Zero-Shot (Nearby Sites)**: Train RF on data from the other 2 locations only, excluding all target-site data entirely. This simulates deploying a correction model to a new field site.
2. **Specialized (Local Data)**: Train RF on local data only (upper bound for comparison).
3. **Pooled (All Sites)**: Train on all 3 locations including the target (best case).
4. **Pooled (Downsampled to N)**: Train on a random sample of the pooled data matching the local dataset size. This controls for the effect of training set volume.

## Results
| Target Location | Training Strategy | Train Size | Corrected RMSE (°C) | Raw NicheMapR (°C) | Improvement (%) |
| --- | --- | --- | --- | --- | --- |
| Ashkelon | Zero-Shot (Nearby Sites) | 9357 | 3.284 | 10.403 | 68.4% |
| Ashkelon | Specialized (Local Data) | 4631 | 1.089 | 10.403 | 89.5% |
| Ashkelon | Pooled (All Sites) | 13988 | 1.088 | 10.403 | 89.5% |
| Ashkelon | Pooled (Downsampled to N) | 4631 | 1.721 | 10.403 | 83.5% |
| Range_24 | Zero-Shot (Nearby Sites) | 9524 | 2.488 | 7.585 | 67.2% |
| Range_24 | Specialized (Local Data) | 4464 | 0.605 | 7.585 | 92.0% |
| Range_24 | Pooled (All Sites) | 13988 | 0.664 | 7.585 | 91.2% |
| Range_24 | Pooled (Downsampled to N) | 4464 | 1.459 | 7.585 | 80.8% |
| Rosh_HaNikra | Zero-Shot (Nearby Sites) | 9095 | 3.615 | 8.552 | 57.7% |
| Rosh_HaNikra | Specialized (Local Data) | 4893 | 1.017 | 8.552 | 88.1% |
| Rosh_HaNikra | Pooled (All Sites) | 13988 | 1.035 | 8.552 | 87.9% |
| Rosh_HaNikra | Pooled (Downsampled to N) | 4893 | 2.303 | 8.552 | 73.1% |

## Visual Summary
![Zero-Shot Transfer Comparison](zero_shot_transfer.png)

## Key Findings

### Zero-Shot Transfer Provides Substantial Correction
Even without any local training data, the zero-shot model reduces NicheMapR error by **58-68%** across all Beach locations. This confirms that the physical feature representation (radiation, humidity, wind speed, temporal encoding) captures generalizable correction patterns that transfer across sites.

### The Gap to Local Models
The zero-shot corrected RMSE (~2.5-3.6°C) is notably higher than locally-trained models (~0.6-1.1°C), indicating that **site-specific physical parameters** (localized albedo, wind blocks, terrain shading) cannot be fully resolved without some local data representation.

### Practical Recommendation
For a new field site where no logger data is available, deploying a zero-shot correction model trained on nearby regional loggers provides a meaningful first-pass correction (**>58% error reduction**) over raw NicheMapR output. Once even a small amount of local logger data becomes available, retraining as a specialized or pooled model will dramatically improve accuracy.

