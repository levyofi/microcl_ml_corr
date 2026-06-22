# Scenario 3: Judean Desert Habitat (Tzeelim) Report

Local correction models trained on two microhabitats (Rock, Bush) at the Tzeelim desert site.
Split: 75% train / 12.5% validation / 12.5% test (random 7-day blocks). Full training set used.

## 1. Example Predictions (120 Hours)

![Desert predictions example](prediction_examples_desert.png)

## 2. Performance at Full Training Data

| Microhabitat | Baseline NicheMapR RMSE (°C) | RF RMSE (°C) | RF Imp (%) | LSTM (2h) RMSE (°C) | LSTM (2h) Imp (%) |
| --- | --- | --- | --- | --- | --- |
| Rock  | 6.536 | 1.657 | 74.6% | 2.156 | 67.0% |
| Bush  | 6.344 | 2.090 | 67.1% | 1.619 | 74.5% |
| **Average** | **6.440** | **1.873** | **70.9%** | **1.888** | **70.7%** |

## 3. Key Takeaway

RF and LSTM perform comparably on average (~71% improvement each), with RF leading on Rock
and LSTM leading on Bush. The NicheMapR baseline error is large (~6.4 °C) due to complex
rock and bush surface energy balance; both models correct it substantially.
To find the minimum number of training days needed, run `learning_curve_example.R`.

---

> **Note on reproducibility:** Results depend on the random 75/12.5/12.5 block split and on the
> random initialisation of the LSTM weights. Re-running the script with a different `SEED` value,
> or on a different machine, will produce slightly different numbers. The direction of the results
> (which model performs better, approximate improvement %) is stable across runs.
