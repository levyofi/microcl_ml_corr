# Scenario 2: Coastal Beach Habitat (Ashkelon 15 m) Report

Local correction model trained on a single beach logger (Ashkelon 15 m).
Split: 75% train / 12.5% validation / 12.5% test (random 7-day blocks). Full training set used.

## 1. Example Predictions (120 Hours)

![Beach predictions example](prediction_examples_beach.png)

## 2. Performance at Full Training Data

| Model | Baseline NicheMapR RMSE (°C) | Corrected RMSE (°C) | Improvement (%) |
| --- | --- | --- | --- |
| RF      | 11.946 | 1.399 | 88.3% |
| LSTM_2h | 11.946 | 2.501 | 79.1% |

## 3. Key Takeaway

RF outperforms LSTM at this training set size. Both achieve large improvements over
the NicheMapR baseline (~79–88%), demonstrating that even a single beach logger
provides sufficient signal to substantially reduce coastal microclimate errors.
To find the minimum number of training days needed, run `learning_curve_example.R`.
