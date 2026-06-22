# Scenario 1: Mediterranean Valley Habitat (Harod) Report

Local correction models trained on three microhabitats (Sun, Shade, Air) at the Harod valley site.
Split: 75% train / 12.5% validation / 12.5% test (random 7-day blocks). Full training set used.

## 1. Example Predictions (120 Hours)

![Valley predictions example](prediction_examples_valley.png)

## 2. Performance at Full Training Data

| Microhabitat | Baseline NicheMapR RMSE (°C) | RF RMSE (°C) | RF Imp (%) | LSTM (2h) RMSE (°C) | LSTM (2h) Imp (%) |
| --- | --- | --- | --- | --- | --- |
| Sun   | 8.169 | 3.484 | 57.3% | 4.083 | 50.0% |
| Shade | 5.530 | 2.828 | 48.9% | 3.220 | 41.8% |
| Air   | 1.792 | 1.466 | 18.2% | 1.341 | 25.1% |
| **Average** | **5.164** | **2.593** | **41.5%** | **2.881** | **38.9%** |

## 3. Key Takeaway

RF and LSTM perform comparably across microhabitats, with RF having a slight edge on Sun and Shade,
while LSTM is marginally better on Air. NicheMapR error is largest for the Sun microhabitat (direct
solar exposure), where both models still achieve ~50% improvement.
To find the minimum number of training days needed, run `learning_curve_example.R`.
