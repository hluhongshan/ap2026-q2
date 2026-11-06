# Advanced Programming 2026 Project Report — Q2

Hao Lu

## Provenance and split disclosure

The work was done in one repository (parent tag `submission`, commit `4521da3`) and split afterwards into one repository per question following the instructor's guidance. Commit hashes changed and the old-to-new commit map is included in `SPLIT_HISTORY.md`. Dates in rewritten commits are the original ones; the split commits are dated when they were made. Shared-file commits appear in more than one repository. `SESSION.md` is the full transcript for all questions and refers to old file names and hashes. Agent work on the split ran with `ap2026-split` as its root.

Section and table numbers below follow the parent report; only this question's sections are included.

### Old-to-new mapping

| Parent | This repository |
|---|---|
| `code/approach3_q2a_mc.py` | `code/approach1_q2a_mc.py` |
| `code/approach4_q2b_deephedge.py` | `code/approach2_q2b_deephedge.py` |
| `results/approach3/` | `results/approach1/` |
| `results/approach4/` | `results/approach2/` |

## 1. Summary

The repository implements Q2 of the Advanced Programming 2026 brief, options A and B. Q2 prices path-dependent payoffs under GBM by Monte Carlo (Q2A) and learns a CVaR-optimal hedge (Q2B).

Q2A passes the Black-Scholes and closed-form checks; the Asian control-variate variance ratio is 308x to 1974x across four futures, above the 5x benchmark. Q2B's deep policy has lower gross and net CVaR than daily Black-Scholes hedging on the brief P&L at zero and at 50 bp cost.

What was not achieved: Q2B covers only IF and uses one training seed. The data are daily, without bid-ask quotes.

## 2. Environment and one-command reproduction

The project uses Python 3.13 under `uv` with pinned dependencies. From the repository root:

```bash
uv sync --python 3.13
source .venv/bin/activate
./reproduce.sh --quick   # tests + compare only
./reproduce.sh           # offline data check, tests, Q2 approaches, compare
python code/check_reproduction.py
```

`./reproduce.sh` runs `data/fetch.py --offline`, the Q2 tests, Q2A Monte Carlo, Q2B deep hedging, and `compare.py`. The offline check verifies the five committed clean artifacts against `data/metadata.json` and reports that all checksums match. The test suite ran 9 tests and passed.

The clean-clone check compares `.cache/clean_clone_full/results` with the committed results. The full clean-clone run completed in 892 s with HTTP_PROXY and HTTPS_PROXY pointed at `http://127.0.0.1:9`. The checker reports 7,692 compared numeric values, all exact, with maximum absolute and relative differences 0.

The sum of committed `fit_seconds` is 807.3 s, about 13.5 minutes; the clean-clone wall time was 14.9 minutes.

## 3. Data

### 3.1 Sources, coverage and snapshots

Table 1: Committed clean artifacts in this repository. Source: `data/metadata.json`.

| artifact | source | rows | retrieved |
|---|---|---|---|
| `data/clean/shibor.csv` | China Shibor | 2398 | 2026-11-04 |
| `data/clean/futures_IF.csv` | Sina main continuous futures IF0 (IF); interface=akshare.futures_main_sina | 2376 | 2026-11-05 |
| `data/clean/futures_RB.csv` | Sina main continuous futures RB0 (RB); interface=akshare.futures_main_sina | 4274 | 2026-11-05 |
| `data/clean/futures_CU.csv` | Sina main continuous futures CU0 (CU); interface=akshare.futures_main_sina | 5310 | 2026-11-05 |
| `data/clean/futures_SC.csv` | Sina main continuous futures SC0 (SC); interface=akshare.futures_main_sina | 2088 | 2026-11-05 |

The four futures series are Sina main continuous contracts (`IF0`, `RB0`, `CU0`, `SC0`). Their cleaned snapshots are committed; the raw downloads were gitignored caches in the parent repository and are not part of this repository. `data/clean/shibor.csv` is the shared rates snapshot. `data/metadata.json` contains only the Q2 artifacts, and `data/fetch.py --offline` verifies every checksum.

## 6. Q2 Option A: Monte Carlo pricing

### 6.1 Model and estimators

Under the brief's GBM, `dS_t = r S_t dt + sigma S_t dW_t`, with `S0` the last close, `K=S0`, `T=1`, `m=252`, barrier `B=1.3 S0`, and `r` the SHIBOR zero at `T`. A futures price has zero drift under Q; the brief specifies drift `r`, so the code follows the brief and flags this as a modelling simplification. `sigma` is the standard deviation of daily log returns times `sqrt(252)` on a five-year window. The payoffs are arithmetic Asian `max(mean(S_{t_i})-K,0)`, discretely monitored up-and-out `max(S_T-K,0) * 1{max S_{t_i}<B}`, cash-or-nothing digital `1{S_T>K}`, and European `max(S_T-K,0)`. All are discounted by `exp(-rT)`.

Estimators are naive Monte Carlo, antithetic `(Z,-Z)` at equal path count, a control variate using the discrete geometric Asian closed form with `ln G ~ N(ln S0 + (r-sigma^2/2)T(m+1)/(2m), sigma^2 T (m+1)(2m+1)/(6m^2))`, and control variate plus antithetic. The control-variate beta is estimated on an independent pilot of 100,000 paths so the estimator is unbiased. European and Asian delta/vega use pathwise estimators; digital and barrier use likelihood-ratio estimators because pathwise is invalid at discontinuities. Finite differences use common random numbers and relative bumps `h in {0.1, 0.01, 0.001}` for spot and sigma. European and digital references are closed form; Asian and barrier references are high-precision control-variate estimates (524,288 paths per dataset, 2,097,152 paths in the IF study). The convergence study uses `N = 2^10` to `2^16`, 30 replications and a `2^22` reference.

### 6.2 Results

Table 14: Q2A calibration. Source: `results/approach1/{IF,CU,RB,SC}/metrics.json`.

| dataset | S0 | sigma | sigma MAD | mu mean | mu log-reg | excess kurtosis | max abs log return | tail-return count |
|---|---|---|---|---|---|---|---|---|
| IF | 4071 | 0.1911 | 0.1426 | -0.03642 | 0.01402 | 10.02 | 0.1056 | 106 |
| CU | 1.062e+05 | 0.1705 | 0.133 | 0.08798 | 0.09923 | 5.109 | 0.06849 | 113 |
| RB | 3008 | 0.1986 | 0.1537 | -0.07184 | -0.09079 | 4.062 | 0.08565 | 131 |
| SC | 681.6 | 0.4197 | 0.3082 | 0.06102 | -0.02652 | 4.974 | 0.1492 | 119 |

Table 15: Q2A prices for every payoff and estimator. Cells are price (standard error), all paths 200,000. Source: `results/approach1/{IF,CU,RB,SC}/metrics.json`.

| dataset | payoff | naive | antithetic | control variate | CV + antithetic |
|---|---|---|---|---|---|
| IF | european | 337.3 (1.18) | 337 (0.9064) | 337.5 (0.6674) | 337.4 (0.6226) |
| IF | asian | 192.7 (0.6542) | 192.6 (0.4904) | 192.9 (0.01666) | 192.8 (0.01653) |
| IF | barrier | 141.6 (0.5798) | 141.6 (0.4844) | 141.6 (0.5754) | 141.5 (0.4587) |
| IF | digital | 0.4854 (0.001102) | 0.4849 (0.0001956) | 0.4855 (0.0008975) | 0.4849 (0.0001949) |
| CU | european | 7950 (27.21) | 7952 (20.63) | 7948 (15.34) | 7943 (14.3) |
| CU | asian | 4533 (15.14) | 4539 (11.25) | 4532 (0.3408) | 4532 (0.3424) |
| CU | barrier | 4160 (15.67) | 4158 (12.63) | 4160 (15.26) | 4160 (12.3) |
| CU | digital | 0.493 (0.001102) | 0.4926 (3.037e-05) | 0.493 (0.0008972) | 0.4926 (3.037e-05) |
| RB | european | 258.6 (0.9081) | 258.6 (0.7012) | 258.7 (0.5166) | 258.9 (0.484) |
| RB | asian | 147.6 (0.503) | 147.4 (0.3804) | 147.6 (0.01313) | 147.6 (0.01305) |
| RB | barrier | 100.3 (0.4224) | 99.93 (0.3582) | 100.3 (0.4206) | 99.87 (0.3357) |
| RB | digital | 0.4842 (0.001102) | 0.4826 (0.0002221) | 0.4842 (0.0008988) | 0.4826 (0.0002211) |
| SC | european | 117.7 (0.4879) | 117.6 (0.4104) | 117.5 (0.284) | 117.5 (0.2778) |
| SC | asian | 67.81 (0.2599) | 67.71 (0.2115) | 67.67 (0.01481) | 67.66 (0.01472) |
| SC | barrier | 5.409 (0.05163) | 5.355 (0.04986) | 5.412 (0.05128) | 5.358 (0.04789) |
| SC | digital | 0.4255 (0.001092) | 0.4244 (0.0005388) | 0.4251 (0.0008934) | 0.4244 (0.000522) |

Table 16: Q2A both-engine agreement z-scores. Source: `results/approach1/{IF,CU,RB,SC}/metrics.json`.

| dataset | European | Asian | barrier | digital |
|---|---|---|---|---|
| IF | 0.6486 | 1.099 | 0.7633 | 1.198 |
| CU | -1.477 | -1.619 | -1.344 | -1.545 |
| RB | -0.1439 | -0.1302 | 0.2969 | 0.932 |
| SC | -0.3891 | -0.2016 | 2.252 | 1.064 |

Table 17: Q2A Greeks versus reference. Source: `results/approach1/{IF,CU,RB,SC}/metrics.json`.

| dataset | payoff | delta | delta ref | delta z | vega | vega ref | vega z |
|---|---|---|---|---|---|---|---|
| IF | european | 0.5682 | 0.568 | 0.1834 | 1598 | 1601 | -0.4397 |
| IF | asian | 0.5397 | 0.5403 | -0.4352 | 926.3 | 927.2 | -0.2409 |
| IF | digital | 0.0005046 | 0.0005053 | -0.4079 | -0.3579 | -0.3523 | -0.5279 |
| IF | barrier | 0.09765 | 0.0803 | 1.182 | -927.9 | -879.3 | -0.5756 |
| CU | european | 0.5679 | 0.5675 | 0.2993 | 4.173e+04 | 4.175e+04 | -0.1115 |
| CU | asian | 0.5425 | 0.5409 | 1.246 | 2.416e+04 | 2.414e+04 | 0.2597 |
| CU | digital | 2.173e-05 | 2.172e-05 | 0.07736 | -0.3995 | -0.3922 | -0.6016 |
| CU | barrier | 0.14 | 0.1304 | 0.5533 | -2.342e+04 | -2.198e+04 | -0.5564 |
| RB | european | 0.5701 | 0.5683 | 1.37 | 1182 | 1182 | 0.01267 |
| RB | asian | 0.5411 | 0.5393 | 1.36 | 684.4 | 686.8 | -0.8341 |
| RB | digital | 0.0006589 | 0.0006581 | 0.3762 | -0.3517 | -0.3407 | -1.078 |
| RB | barrier | 0.06839 | 0.07094 | -0.1841 | -744.4 | -618 | -2.153 |
| SC | european | 0.5982 | 0.5965 | 1.016 | 264.2 | 263.9 | 0.1735 |
| SC | asian | 0.5482 | 0.5473 | 0.6066 | 154.2 | 153.7 | 0.6143 |
| SC | digital | 0.001358 | 0.001353 | 0.9338 | -0.228 | -0.2254 | -0.6084 |
| SC | barrier | -0.0139 | -0.01232 | -0.4994 | -29.23 | -29.93 | 0.2311 |

Table 18: Q2A full finite-difference bias and variance. Source: `results/approach1/IF_study/metrics.json`.

| payoff | greek | bump | mean | bias | variance | RMSE | z |
|---|---|---|---|---|---|---|---|
| european | delta | 0.1 | 0.5616 | -0.006336 | 2.938e-05 | 0.008279 | -6.402 |
| european | delta | 0.01 | 0.5679 | -0.0001023 | 4.043e-05 | 0.006252 | -0.08816 |
| european | delta | 0.001 | 0.5681 | 8.182e-05 | 3.974e-05 | 0.006199 | 0.07109 |
| asian | delta | 0.1 | 0.5308 | -0.008875 | 2.215e-05 | 0.01001 | -9.87 |
| asian | delta | 0.01 | 0.5391 | -0.0005191 | 3.302e-05 | 0.005674 | -0.4797 |
| asian | delta | 0.001 | 0.5391 | -0.0005174 | 3.534e-05 | 0.005868 | -0.4631 |
| barrier | delta | 0.1 | 0.0744 | -0.0104 | 2.998e-05 | 0.01171 | -3.354 |
| barrier | delta | 0.01 | 0.09012 | 0.005317 | 0.0004428 | 0.02136 | 1.1 |
| barrier | delta | 0.001 | 0.08003 | -0.00478 | 0.005104 | 0.0704 | -0.3575 |
| digital | delta | 0.1 | 0.0004832 | -2.211e-05 | 5.21e-11 | 2.322e-05 | -16.78 |
| digital | delta | 0.01 | 0.000506 | 6.842e-07 | 1.326e-09 | 3.58e-05 | 0.1029 |
| digital | delta | 0.001 | 0.000527 | 2.172e-05 | 6.284e-09 | 8.091e-05 | 1.5 |
| european | vega | 0.1 | 1605 | 4.046 | 934.2 | 30.32 | 0.725 |
| european | vega | 0.01 | 1605 | 4.053 | 931 | 30.27 | 0.7275 |
| european | vega | 0.001 | 1605 | 4.047 | 931.7 | 30.28 | 0.7262 |
| asian | vega | 0.1 | 928.4 | 2.246 | 291.3 | 16.93 | 0.6989 |
| asian | vega | 0.01 | 928.5 | 2.267 | 291.1 | 16.93 | 0.7057 |
| asian | vega | 0.001 | 928.5 | 2.277 | 291.2 | 16.93 | 0.7087 |
| barrier | vega | 0.1 | -875.9 | -4.254 | 3689 | 59.87 | -0.211 |
| barrier | vega | 0.01 | -882.5 | -10.9 | 2.978e+04 | 170 | -0.3051 |
| barrier | vega | 0.001 | -810.4 | 61.26 | 4.605e+05 | 670 | 0.4899 |
| digital | vega | 0.1 | -0.3626 | -0.01027 | 0.001128 | 0.03458 | -1.675 |
| digital | vega | 0.01 | -0.3736 | -0.02129 | 0.01481 | 0.1215 | -0.9579 |
| digital | vega | 0.001 | -0.3883 | -0.03598 | 0.1004 | 0.3135 | -0.622 |

At `h=0.1` all four delta estimators fail the z-test: European -6.40, Asian -9.87, barrier -3.35 and digital -16.78. At `h=0.01` and `h=0.001` all delta z-scores are within plus or minus two. Vega z-scores are noisier for the barrier and digital, as expected for discontinuous payoffs.

Table 19: Q2A convergence slopes. Source: `results/approach1/IF_study/metrics.json`.

| payoff | naive | antithetic | control variate | CV + antithetic |
|---|---|---|---|---|
| european | -0.4628 | -0.5224 | -0.4948 | -0.5249 |
| asian | -0.4745 | -0.5164 | -0.5079 | -0.4601 |
| barrier | -0.494 | -0.5198 | -0.4936 | -0.5064 |
| digital | -0.4839 | -0.5361 | -0.4718 | -0.5368 |

The complete pricing slope range is -0.4601 to -0.5368. Greek convergence slopes are reported separately in Table 19b; they range from -0.4427 to -0.4959.

Table 19b: Q2A Greek convergence slopes. Source: `results/approach1/IF_study/metrics.json`.

| payoff | delta slope | vega slope |
|---|---|---|
| european | -0.4762 | -0.4644 |
| asian | -0.4668 | -0.4908 |
| barrier | -0.4427 | -0.4959 |
| digital | -0.4643 | -0.4703 |

Table 20: Q2A timing. Source: `results/approach1/IF_study/metrics.json`.

| paths | monitoring dates | loop seconds | vectorized seconds | speedup | loop paths/s | vectorized paths/s |
|---|---|---|---|---|---|---|
| 2,000 | 252 | 0.09895 | 0.003752 | 26.37 | 2.021e+04 | 5.33e+05 |

The line profile excerpt (source: `results/approach1/line_profile.txt`) shows where the loop spends its time:

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
   414    126500   32700000.0    258.5     17.9          for j in range(m):
   415    126000   49282000.0    391.1     26.9              log_s += drift + vol * z[i, j]
   416    126000   33841000.0    268.6     18.5              value = math.exp(log_s)
   417    126000   32671000.0    259.3     17.8              sum_s += value
   418    126000   30840000.0    244.8     16.8              if value > max_s:
```

The digital antithetic ratio is strongly dataset-dependent. Table 21 reports the drift `r`, volatility `sigma`, the derived quantity `r - sigma^2/2` and the ratio. Source: `results/approach1/{IF,CU,RB,SC}/metrics.json`.

| dataset | r | sigma | r - sigma^2/2 (derived) | digital antithetic ratio |
|---|---|---|---|---|
| IF | 0.01446 | 0.1911 | -0.003795 | 31.73 |
| CU | 0.01446 | 0.1705 | -7.22e-05 | 1316 |
| RB | 0.01446 | 0.1986 | -0.005251 | 24.61 |
| SC | 0.01446 | 0.4197 | -0.07363 | 4.105 |

Hypothesis, supported by the ordering but not proved: for an ATM digital, when `r - sigma^2/2` is near zero, a `Z, -Z` pair almost always has exactly one in-the-money leg, so the pair mean is nearly constant. CU has the value closest to zero and the largest ratio; SC has the most negative value and the smallest ratio. This is an explanation of the ordering, not a tested theorem.

### 6.3 Discussion and limitations

The Asian control variate is very effective because the geometric Asian is strongly correlated with the arithmetic payoff. The barrier control variate is weak: control-variate ratios are 1.008 to 1.055, while CV+antithetic ratios are 1.162 to 1.624. Finite differences are noisy for discontinuous payoffs: barrier vega RMSE grows from 59.87 at `h=0.1` to 670.0 at `h=0.001`, so smaller bumps do not help. The roll-jump counts in Table 14 are tail-return counts above the reported threshold, not identified roll dates; the main-contract series carry no roll flags. The calibration uses a five-year window and is sensitive to the large returns retained in the sample. The GBM drift is `r` as specified, not the zero drift of a futures price under Q.

## 7. Q2 Option B: deep hedging

### 7.1 Model

The hedge is a European ATM call on IF with `S0=100`, `K=100`, `T=0.25`, `m=63`, `sigma=0.1911` (the Q2A five-year IF calibration) and `r=0.01417` (SHIBOR zero at `T`). The policy is a small MLP with two hidden layers of 32 units mapping `(S_t/S0, t/T)` to a delta in `[0,1]` through a sigmoid; it does not see the previous position. The objective is empirical CVaR at 0.95 of `-V_T`, the mean of the worst 5% of losses per batch. Training uses `2^17` paths, batches of `2^12`, 100 epochs, Adam at learning rate 0.005, and one policy per cost `kappa in {0,10,50}` bp. The same network initialization is used for every kappa. Validation uses three independent seeds with `2^17` paths each; training uses one seed (42).

The brief P&L is `V_T = sum_t exp(r(T-t)) D_t (S_{t+1}-S_t) - kappa sum_t |D_t-D_{t-1}| S_t - payoff`, with `D_{-1}=0` and no terminal unwind. The self-financing P&L is `V_T = sum_t exp(r(T-t-dt)) D_t (S_{t+1}-exp(r dt) S_t) - kappa sum_t exp(r(T-t)) |D_t-D_{t-1}| S_t - payoff`. The brief formula omits the financing cost of the stock position. Training uses the brief P&L; evaluation reports both. Net CVaR is CVaR minus the future value of the Black-Scholes premium. Benchmarks are Black-Scholes deltas rebalanced every 1, 5 and 21 days at the same kappas, on the same validation paths.

### 7.2 Results

Table 22: Q2B full results for every strategy, kappa and P&L definition. Source: `results/approach2/IF/metrics.json`.

| method | rebalance d | policy kappa bp | eval kappa bp | P&L | CVaR | net CVaR | variance | E[P&L] | turnover |
|---|---|---|---|---|---|---|---|---|---|
| BS | 1 | n/a | 0 | brief | 4.781 | 0.7838 | 0.1815 | -3.808 | 3.04 |
| BS | 1 | n/a | 0 | selffin | 4.961 | 0.9634 | 0.1733 | -3.998 | 3.04 |
| BS | 1 | n/a | 10 | brief | 5.198 | 1.201 | 0.199 | -4.112 | 3.04 |
| BS | 1 | n/a | 10 | selffin | 5.381 | 1.383 | 0.1904 | -4.302 | 3.04 |
| BS | 1 | n/a | 50 | brief | 6.971 | 2.974 | 0.4371 | -5.328 | 3.04 |
| BS | 1 | n/a | 50 | selffin | 7.16 | 3.163 | 0.4273 | -5.521 | 3.04 |
| BS | 5 | n/a | 0 | brief | 5.883 | 1.885 | 0.7867 | -3.807 | 1.645 |
| BS | 5 | n/a | 0 | selffin | 6.069 | 2.072 | 0.7814 | -3.997 | 1.645 |
| BS | 5 | n/a | 10 | brief | 6.101 | 2.104 | 0.8066 | -3.971 | 1.645 |
| BS | 5 | n/a | 10 | selffin | 6.289 | 2.291 | 0.8012 | -4.162 | 1.645 |
| BS | 5 | n/a | 50 | brief | 6.998 | 3.001 | 0.9166 | -4.629 | 1.645 |
| BS | 5 | n/a | 50 | selffin | 7.188 | 3.191 | 0.9111 | -4.821 | 1.645 |
| BS | 21 | n/a | 0 | brief | 8.101 | 4.104 | 3.112 | -3.805 | 0.9313 |
| BS | 21 | n/a | 0 | selffin | 8.296 | 4.299 | 3.114 | -3.995 | 0.9313 |
| BS | 21 | n/a | 10 | brief | 8.211 | 4.214 | 3.145 | -3.898 | 0.9313 |
| BS | 21 | n/a | 10 | selffin | 8.407 | 4.41 | 3.146 | -4.088 | 0.9313 |
| BS | 21 | n/a | 50 | brief | 8.655 | 4.658 | 3.28 | -4.271 | 0.9313 |
| BS | 21 | n/a | 50 | selffin | 8.853 | 4.856 | 3.282 | -4.462 | 0.9313 |
| deep | n/a | 0 | 0 | brief | 4.679 | 0.6817 | 0.2323 | -3.809 | 2.848 |
| deep | n/a | 0 | 0 | selffin | 4.888 | 0.891 | 0.237 | -3.997 | 2.848 |
| deep | n/a | 10 | 10 | brief | 4.987 | 0.9894 | 0.25 | -4.085 | 2.77 |
| deep | n/a | 10 | 10 | selffin | 5.176 | 1.179 | 0.2492 | -4.275 | 2.77 |
| deep | n/a | 50 | 50 | brief | 6.083 | 2.085 | 0.3018 | -5.072 | 2.523 |
| deep | n/a | 50 | 50 | selffin | 6.331 | 2.334 | 0.3178 | -5.261 | 2.523 |

Table 23: Q2B 3x3 cross-evaluation. Source: `results/approach2/IF/metrics.json`.

| policy kappa bp | eval kappa bp | CVaR brief | net CVaR brief | CVaR selffin | net CVaR selffin | turnover |
|---|---|---|---|---|---|---|
| 0 | 0 | 4.679 | 0.6817 | 4.888 | 0.891 | 2.848 |
| 0 | 10 | 5.008 | 1.011 | 5.209 | 1.211 | 2.848 |
| 0 | 50 | 6.533 | 2.536 | 6.724 | 2.726 | 2.848 |
| 10 | 0 | 4.717 | 0.7195 | 4.906 | 0.9084 | 2.77 |
| 10 | 10 | 4.987 | 0.9894 | 5.176 | 1.179 | 2.77 |
| 10 | 50 | 6.379 | 2.382 | 6.568 | 2.571 | 2.77 |
| 50 | 0 | 5.06 | 1.063 | 5.31 | 1.313 | 2.523 |
| 50 | 10 | 5.229 | 1.231 | 5.484 | 1.486 | 2.523 |
| 50 | 50 | 6.083 | 2.085 | 6.331 | 2.334 | 2.523 |

Table 24: Q2B deep policy versus daily Black-Scholes on the brief P&L. Source: `results/approach2/IF/metrics.json`.

| kappa bp | BS daily gross CVaR | deep gross CVaR | BS daily net CVaR | deep net CVaR |
|---|---|---|---|---|
| 0 | 4.781 | 4.679 | 0.7838 | 0.6817 |
| 50 | 6.971 | 6.083 | 2.974 | 2.085 |

On the brief P&L, the deep policy lowers gross CVaR by 0.10 at kappa 0 and by 0.89 at kappa 50; net CVaR falls by the same amounts because the premium is common. The deep policy has higher variance than daily BS at kappa 0 (0.2323 vs 0.1815 brief; 0.2370 vs 0.1733 self-financing) and lower variance at kappa 50 (0.3018 vs 0.4371 brief; 0.3178 vs 0.4273 self-financing). A plausible cause of the CVaR gain is that the policy optimizes CVaR directly rather than following a delta rule, but the experiment does not isolate optimization from the state-dependent delta; that is a hypothesis.

The brief-minus-self-financing CVaR gap is negative everywhere: -0.1796 for BS daily at kappa 0, -0.2094 for the deep policy at kappa 0, and -0.2486 for the deep policy at kappa 50; the mean across all cells is -0.1960. The negative sign means the brief formula reports lower CVaR than the self-financing formula. Training CVaR falls from 11.01 to 4.70 (kappa 0), 11.08 to 4.98 (kappa 10) and 11.32 to 6.13 (kappa 50).

Figures: `results/approach2/IF/training.html` and `results/approach2/IF/policy.html`.

### 7.3 Discussion and limitations

The deep hedge improves CVaR over daily BS on the brief basis at all three costs, with lower turnover at higher cost. The gap is systematic and is larger for the deep policy at high cost. A plausible explanation, not tested here, is that the omitted stock financing scales with the position and the position is held through the path. The training used one seed only; three validation seeds reduce evaluation noise but do not establish training robustness. The study covers only IF, uses no terminal unwind cost, and the policy cannot represent a no-trade region because the previous position is not an input.

## 10. Indicative benchmarks

The brief file is held outside the repository. The target column below is quoted from the project brief, not from `AGENTS.md` or a results file. Every number in the reported-result column is from the metrics files named in the caption.

Table 38: Indicative Q2 benchmarks. Source: `results/approach1/{IF,CU,RB,SC}/metrics.json`, `results/approach1/IF_study/metrics.json`, `results/approach2/IF/metrics.json`; target column from the brief/instructions.

| benchmark | target | reported result | met? |
|---|---|---|---|
| Q2 Asian naive-vs-control-variate variance reduction | factor >= 5 | 308.3x to 1974x (CV), 312.0x to 1955x (CV+anti) | met |
| Q2 pathwise/FD delta agreement | within reported standard errors | h=0.01 and 0.001 pass; h=0.1 fails for European, Asian, barrier and digital (digital z=-16.78) | partly met |

The benchmark table contains only the brief's indicative targets.

## 11. Differences from the brief

1. **Hedging P&L.** The printed brief P&L omits the financing cost of the stock position, so it is not exactly self-financing. The brief-minus-self-financing CVaR gap is -0.1796 for daily BS at kappa 0 and -0.2094 for the deep policy at kappa 0; the mean across all cells is -0.1960. The negative sign means the brief formula reports lower CVaR than the self-financing formula. Source: `results/approach2/IF/metrics.json`.
2. **GBM drift.** The brief specifies drift `r`; the code follows the brief. A futures price has zero drift under Q, so this is a modelling simplification, recorded in `params.drift_note`. Source: `results/approach1/IF/metrics.json`.
3. **Q2B scope.** Only IF is used, with one training seed and no terminal unwind; the brief permits this but it limits generalisation. Source: `results/approach2/IF/metrics.json`.
4. **Individual submission.** This is an individual submission rather than a group submission.
5. **AI-drafted report.** The report was drafted by AI rather than written by a group, with instructor approval as stated by the author in Section 14.
6. **FastboxData not used.** FastboxData was not used as a data source.

## 12. Limitations

- Main-contract futures series have no roll flags. The fields named `roll_jump` in `metrics.json` are tail-return counts (IF 106, CU 113, RB 131, SC 119), not identified roll dates.
- Q2B used one training seed, only IF, no terminal unwind cost, and no previous position in the policy inputs; it cannot represent a no-trade region.

## 13. Session integrity

Known writes outside the repository root, from the session record:

- `/tmp/q1_online.log`, `/tmp/q1_online_resume.log`, `/tmp/q1_online_resume2.log`, `/tmp/q1_expiry_fix.log`: logs from the early Q1 data downloads.
- `/tmp/q1a_close.log`: log from the early Q1A close run (still present; not deleted).
- `/tmp/offline_check.log`, `/tmp/offline_check2.log`, `/tmp/uv_sync_final.log`: reproduction and environment checks.
- `~/.cache/uv`: possible writes from early `uv run` calls before `UV_CACHE_DIR` was pinned to the repository.
- `~/.local/share/opencode/tool-output/`: automatic tool-output files written by the harness.

Tool setup and other session facts:

- Before the first working session, the desktop app and CLI conflicted over one database, so the file in `~/.local/share/opencode` was moved by hand to a backup folder. This was tool setup, not a project change.
- An accidental one-word prompt was sent to a free model (session 01); it ran read-only commands and was stopped.
- Free quota ran out on 2026-11-04, so the session continued on OpenCode Go with the same model family.
- During reproduction the agent copied the uv-managed Python into `.cache` with read access to `~/.local/share/uv`, and deleted its own temporary clone under `.cache` without a separate approval.
- Claude also renamed two misnamed session files and read repository files, on the user's computer, without changing any results.

The report step itself wrote only `report.md` and a verification script under `.cache/`; it did not change code, data or results. The working tree already contained a modified `results/comparison.html` on entry; it was not touched.

The split work itself ran with `ap2026-split` as its root; `../ap2026` was read-only, and the split repositories were created afterwards from parent tag `submission`.

## 14. Statement of work

This is an individual submission by Hao Lu. The instructor approved individual completion with AI assistance. opencode with DeepSeek V4.1 Flash wrote the code, ran the experiments and drafted this report. Claude, in a separate chat that is not part of SESSION.md, drafted most of the prompts, proposed many methodology decisions and corrections, and reviewed outputs. Hao Lu directed the work, approved each step, and owns the final submission after reviewing it. The author does not claim to have independently derived the mathematics; the derivations above are a record of the implemented methods and their sources. The author has read this report.

## 15. Causal claims

Table 39: Prose claims about causes and their status. Source: `results/approach1/IF_study/metrics.json`, `results/approach2/IF/metrics.json`, and the report text.

| claim | status |
|---|---|
| The digital antithetic ratio ordering is explained by `r - sigma^2/2` near zero | hypothesis; supported by the ordering across datasets, not proved |
| The deep hedge CVaR gain is caused by direct CVaR optimization | hypothesis; optimization is not isolated from the state-dependent delta |
| The brief-minus-self-financing gap is caused by the omitted stock financing | tested by the two P&L formulas; the gap is a direct consequence of the formula difference |
| The brief-minus-self-financing gap is larger at high cost because omitted financing scales with the position | hypothesis; the pattern is visible in the table, but the scaling was not isolated |
| The Q2 clean-clone comparison is exact | tested by the checker: 7,692 shared numeric values, zero differences |

## 16. Result files

Every number in the tables is from one of the following result files unless the table caption says otherwise: `results/approach1/IF/metrics.json`, `results/approach1/CU/metrics.json`, `results/approach1/RB/metrics.json`, `results/approach1/SC/metrics.json`, `results/approach1/IF_study/metrics.json`, `results/approach2/IF/metrics.json`. Metadata and row counts come from `data/metadata.json`. The clean-clone totals in Section 2 are from the `code/check_reproduction.py` output, and the session-integrity facts are from the session record. Former `approach3` is now `approach1`; former `approach4` is now `approach2`.
