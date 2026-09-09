# Preprint — main results

## Table 1 — FLORES causal intervention

| model | pooling | normal final R@1 | language final R@1 | PCA final R@1 | random final R@1 | language - PCA | CI 95 % |
|---|---|---:|---:|---:|---:|---:|---|
| Tri-0.5B | mean | 0.124 | 0.325 | 0.148 | 0.122 | 0.177 | [0.160; 0.193] |
| Tri-0.5B | last_token | 0.232 | 0.310 | 0.215 | 0.227 | 0.096 | [0.079; 0.112] |
| Qwen2.5-0.5B | mean | 0.043 | 0.706 | 0.686 | 0.042 | 0.020 | [0.014; 0.026] |
| Qwen2.5-0.5B | last_token | 0.026 | 0.366 | 0.357 | 0.026 | 0.010 | [0.003; 0.015] |

## Table 2 — MASSIVE

| model | pooling | normal target accuracy | language target accuracy | PCA target accuracy | random target accuracy | EN accuracy normal | EN accuracy language |
|---|---|---:|---:|---:|---:|---:|---:|
| Tri-0.5B | mean | 0.297 | 0.444 | 0.438 | 0.286 | 0.773 | 0.784 |
| Tri-0.5B | last_token | 0.071 | 0.367 | 0.448 | 0.072 | 0.719 | 0.739 |
| Qwen2.5-0.5B | mean | 0.264 | 0.420 | 0.424 | 0.266 | 0.789 | 0.791 |
| Qwen2.5-0.5B | last_token | 0.324 | 0.365 | 0.364 | 0.315 | 0.758 | 0.764 |

## Table 3 — Language modeling cost

| model | language | normal NLL | language NLL | PCA NLL | delta language | delta PCA |
|---|---|---:|---:|---:|---:|---:|
| Tri-0.5B | EN | 3.816 | 5.009 | 7.318 | 1.193 | 3.502 |
| Tri-0.5B | KO | 4.815 | 11.848 | 6.823 | 7.033 | 2.008 |
| Tri-0.5B | JA | 4.473 | 9.531 | 5.923 | 5.059 | 1.450 |
| Tri-0.5B | ZH | 4.596 | 7.074 | 6.249 | 2.478 | 1.652 |
| Qwen2.5-0.5B | EN | 3.450 | 3.657 | 3.594 | 0.207 | 0.144 |
| Qwen2.5-0.5B | KO | 3.238 | 6.547 | 6.853 | 3.309 | 3.614 |
| Qwen2.5-0.5B | JA | 3.364 | 6.033 | 6.895 | 2.670 | 3.531 |
| Qwen2.5-0.5B | ZH | 3.868 | 6.185 | 7.555 | 2.316 | 3.687 |

Target accuracy is the unweighted mean over KO, JA, and ZH. Random values are means over five reproducible orthonormal bases. All confidence intervals use 1,000 paired bootstrap resamples.
