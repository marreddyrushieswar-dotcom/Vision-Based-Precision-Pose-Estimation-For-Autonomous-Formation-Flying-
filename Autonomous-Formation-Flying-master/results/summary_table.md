Mean ± std over 10 Monte-Carlo runs of the 10 s approach scenario.

| Run | Method | Position RMSE [m] | Attitude error [deg] |
|---|---|---|---|
| paper: inverse-MSE likelihood | ANN only | 6.02 ± 0.03 | 5.04 ± 0.03 |
| paper: inverse-MSE likelihood | ANN only (softmax mean) | 3.09 ± 0.01 | 2.85 ± 0.01 |
| paper: inverse-MSE likelihood | PF N=1000, a=0.9 every 10 cycles | 4.39 ± 1.92 | 4.16 ± 1.54 |
| paper: inverse-MSE likelihood | PF N=5000, a=0.9 every cycle | 3.87 ± 1.42 | 3.65 ± 1.10 |
| Gaussian likelihood + LM refine | ANN only | 6.02 ± 0.03 | 5.04 ± 0.03 |
| Gaussian likelihood + LM refine | ANN only (softmax mean) | 3.09 ± 0.01 | 2.85 ± 0.01 |
| Gaussian likelihood + LM refine | ANN + LM refine | 2.54 ± 0.01 | 1.98 ± 0.00 |
| Gaussian likelihood + LM refine | PF N=1000, a=0.9 every 10 cycles | 2.76 ± 1.09 | 4.73 ± 0.49 |
| Gaussian likelihood + LM refine | PF N=1000, a=0.9 every 10 cycles + LM refine | 2.54 ± 0.01 | 1.98 ± 0.00 |
| Gaussian likelihood + LM refine | PF N=5000, a=0.9 every cycle | 2.05 ± 0.54 | 3.67 ± 0.06 |
| Gaussian likelihood + LM refine | PF N=5000, a=0.9 every cycle + LM refine | 2.54 ± 0.01 | 1.98 ± 0.00 |
| Gaussian likelihood + softmax proposal | ANN only | 6.02 ± 0.03 | 5.04 ± 0.03 |
| Gaussian likelihood + softmax proposal | ANN only (softmax mean) | 3.09 ± 0.01 | 2.85 ± 0.01 |
| Gaussian likelihood + softmax proposal | PF N=1000, a=0.9 every 10 cycles | 2.81 ± 0.40 | 4.61 ± 0.24 |
| Gaussian likelihood + softmax proposal | PF N=5000, a=0.9 every cycle | 2.07 ± 0.08 | 3.51 ± 0.04 |
| Gaussian likelihood + LM refine, known LOS | ANN only | 4.12 ± 0.07 | 5.04 ± 0.03 |
| Gaussian likelihood + LM refine, known LOS | ANN only (softmax mean) | 1.62 ± 0.01 | 2.85 ± 0.01 |
| Gaussian likelihood + LM refine, known LOS | ANN + LM refine | 0.31 ± 0.01 | 0.26 ± 0.01 |
| Gaussian likelihood + LM refine, known LOS | PF N=1000, a=0.9 every 10 cycles | 1.34 ± 0.32 | 4.12 ± 0.04 |
| Gaussian likelihood + LM refine, known LOS | PF N=1000, a=0.9 every 10 cycles + LM refine | 0.31 ± 0.01 | 0.26 ± 0.01 |
| Gaussian likelihood + LM refine, known LOS | PF N=5000, a=0.9 every cycle | 1.50 ± 0.18 | 3.21 ± 0.03 |
| Gaussian likelihood + LM refine, known LOS | PF N=5000, a=0.9 every cycle + LM refine | 0.31 ± 0.01 | 0.26 ± 0.01 |
| Gaussian (sigma 1 px) + softmax proposal, known LOS | ANN only | 4.12 ± 0.07 | 5.04 ± 0.03 |
| Gaussian (sigma 1 px) + softmax proposal, known LOS | ANN only (softmax mean) | 1.62 ± 0.01 | 2.85 ± 0.01 |
| Gaussian (sigma 1 px) + softmax proposal, known LOS | PF N=1000, a=0.9 every 10 cycles | 0.39 ± 0.09 | 1.50 ± 0.05 |
| Gaussian (sigma 1 px) + softmax proposal, known LOS | PF N=5000, a=0.9 every cycle | 0.32 ± 0.07 | 1.43 ± 0.03 |
