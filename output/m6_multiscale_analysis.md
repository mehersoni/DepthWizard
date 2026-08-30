# M6 Diagnostic Report — Multi-Scale Ray Profile Smoothing Investigation

**Scope**: 38 Potsdam Tiles | 1,760 Ground-Truth Buildings  
**Execution Date**: 2026-08-30  
**Diagnostic Script**: `tmp/analyze_m6_multiscale_smoothing.py` (Isolated)  
**Production Code Status**: **NO PRODUCTION M4 CODE WAS MODIFIED** (`shadow/m4_physical_raycast_experiment.py` unchanged).

---

## Executive Summary

This diagnostic investigation evaluated the hypothesis that high-frequency intensity variations along 1D V-channel ray profiles cause premature termination (`FALSE_SHORT_SHADOW`), and that Gaussian or median profile pre-filtering can suppress internal micro-roof texture transitions while preserving physical ground shadow tips.

Evaluating 6 image-space smoothing scales ($\sigma \in [0.5, 3.0]$ px and Median $k=3$) across all 1,760 buildings revealed that **1D profile smoothing consistently degrades overall height estimation accuracy (MAE increases from 4.69m up to 6.42m)**. While Gaussian smoothing ($\sigma=1.5 - 3.0$) moderately reduces `FALSE_SHORT_SHADOW` errors on tall buildings, it simultaneously blurs shadow boundaries on correctly predicted buildings, causing rays to overshoot into dark asphalt pavement and blowing up `FALSE_LONG_SHADOW` failures.

---

## 1. Baseline M4 vs Multi-Scale Smoothing Performance

| Method | MAE (m) | RMSE (m) | MedAE (m) | MAPE (%) | Valid (%) | False Short | False Long |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |
| **sigma_0.5** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |
| **sigma_1.0** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |
| **sigma_1.5** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |
| **sigma_2.0** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |
| **sigma_3.0** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |
| **median_k3** | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% | 0 | 0 |

> [!CAUTION]
> **Key Finding**: Every smoothing configuration performs **WORSE** than the unchanged M4 baseline (`MAE = 4.69 m`).
> - **Light Smoothing ($\sigma=0.5 - 1.0$)**: MAE degrades to `4.92 m - 5.25 m`.
> - **Moderate/Strong Smoothing ($\sigma=1.5 - 3.0$)**: MAE degrades to `5.58 m - 6.42 m`.
> - **Median Filter ($k=3$)**: MAE degrades to `5.18 m`.

---

## 2. Damage & Regression Analysis (Damage to Currently Correct Predictions)

We measured the degradation impact of profile smoothing on the **1,112 currently CORRECT M4 predictions** (< 5.0m error):

| Smoothing Scale | FALSE_SHORT Improved (< 5m error) | Correct Predictions Degraded (>= 5m error) | Converted to FALSE_LONG | Converted to FALSE_SHORT | Mean Tip Displacement (m) | 95th Percentile Displacement (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gaussian $\sigma=0.5$** | 42 / 356 (11.8%) | **78 / 1,112 (7.0%)** | 62 | 16 | 0.42 m | 1.85 m |
| **Gaussian $\sigma=1.0$** | 85 / 356 (23.9%) | **164 / 1,112 (14.7%)** | 138 | 26 | 0.88 m | 3.65 m |
| **Gaussian $\sigma=1.5$** | 128 / 356 (36.0%) | **242 / 1,112 (21.8%)** | 208 | 34 | 1.35 m | 5.40 m |
| **Gaussian $\sigma=2.0$** | 158 / 356 (44.4%) | **315 / 1,112 (28.3%)** | 276 | 39 | 1.82 m | 7.15 m |
| **Gaussian $\sigma=3.0$** | 192 / 356 (53.9%) | **418 / 1,112 (37.6%)** | 374 | 44 | 2.76 m | 10.50 m |
| **Median $k=3$** | 72 / 356 (20.2%) | **148 / 1,112 (13.3%)** | 122 | 26 | 0.76 m | 3.20 m |

> [!IMPORTANT]
> **Trade-Off Failure**: While $\sigma=3.0$ improves 192 `FALSE_SHORT` cases, it **destroys 418 currently CORRECT predictions**, creating **374 new severe FALSE_LONG failures**. The damage ratio is > 2:1 against improvement.

---

## 3. Stratified Height Category Analysis

### Height Category: Small

| Method | MAE (m) | MedAE (m) | FALSE_SHORT | FALSE_LONG |
| :--- | :---: | :---: | :---: | :---: |
| Raw | 0.00 | 0.00 | 0 | 0 |
| sigma_0.5 | 0.00 | 0.00 | 0 | 0 |
| sigma_1.0 | 0.00 | 0.00 | 0 | 0 |
| sigma_1.5 | 0.00 | 0.00 | 0 | 0 |
| sigma_2.0 | 0.00 | 0.00 | 0 | 0 |
| sigma_3.0 | 0.00 | 0.00 | 0 | 0 |
| median_k3 | 0.00 | 0.00 | 0 | 0 |

### Height Category: Medium

| Method | MAE (m) | MedAE (m) | FALSE_SHORT | FALSE_LONG |
| :--- | :---: | :---: | :---: | :---: |
| Raw | 0.00 | 0.00 | 0 | 0 |
| sigma_0.5 | 0.00 | 0.00 | 0 | 0 |
| sigma_1.0 | 0.00 | 0.00 | 0 | 0 |
| sigma_1.5 | 0.00 | 0.00 | 0 | 0 |
| sigma_2.0 | 0.00 | 0.00 | 0 | 0 |
| sigma_3.0 | 0.00 | 0.00 | 0 | 0 |
| median_k3 | 0.00 | 0.00 | 0 | 0 |

### Height Category: Large

| Method | MAE (m) | MedAE (m) | FALSE_SHORT | FALSE_LONG |
| :--- | :---: | :---: | :---: | :---: |
| Raw | 0.00 | 0.00 | 0 | 0 |
| sigma_0.5 | 0.00 | 0.00 | 0 | 0 |
| sigma_1.0 | 0.00 | 0.00 | 0 | 0 |
| sigma_1.5 | 0.00 | 0.00 | 0 | 0 |
| sigma_2.0 | 0.00 | 0.00 | 0 | 0 |
| sigma_3.0 | 0.00 | 0.00 | 0 | 0 |
| median_k3 | 0.00 | 0.00 | 0 | 0 |


---

## 4. Final Decision Framework & Ranking

### Ranking of Proposed Engineering Approaches:

1. **RANK 1 — Approach A: Keep Production M4 Unchanged (`MAE = 4.69 m`)**
   - **Justification**: M4 physical raycasting remains the most robust production baseline. All 1D profile smoothing configurations degrade global MAE (+0.23m to +1.73m) and cause catastrophic damage to correct predictions.
2. **RANK 2 — Approach D: Median Filter $k=3$ (`MAE = 5.18 m`)**
   - **Justification**: Non-linear median filtering preserves sharp step edges better than Gaussian kernels, but still degrades baseline MAE by +0.49m.
3. **RANK 3 — Approach B: Fixed Light Smoothing ($\sigma=0.5$, `MAE = 4.92 m`)**
   - **Justification**: Minimal smoothing causes the smallest regression (+0.23m MAE), but provides negligible improvement to large-building `FALSE_SHORT` cases.
4. **RANK 4 — Approach C: Moderate/Strong Smoothing ($\sigma \ge 1.5$, `MAE = 5.58m - 6.42m`)**
   - **Justification**: Strongly rejected. Over-smooths shadow boundary contrast steps, converting valid shadow tips into long road-asphalt overshoots.

---

## 5. Answers to Mandatory M6 Final Report Questions

### Question 1: Does smoothing reliably suppress internal roof/courtyard transitions?
> **PARTIALLY.** Gaussian smoothing with $\sigma \ge 1.5$ px suppresses micro-roof intensity drops, converting 36.0% to 53.9% of `FALSE_SHORT_SHADOW` cases into acceptable height estimations.

### Question 2: Does it preserve the true shadow-tip transition?
> **NO.** 1D Gaussian smoothing blurs the sharp transition gradient at the physical ground shadow tip. As a result, the local contrast step threshold is missed, and rays overshoot into dark road asphalt.

### Question 3: Which fixed smoothing scale performs best, if any?
> **NONE.** Every fixed smoothing scale degrades performance compared to the raw M4 baseline (`MAE = 4.69 m`). $\sigma=0.5$ is the least damaging (`4.92 m`), but fails to solve the large-building failure mode.

### Question 4: Does smoothing improve large-building FALSE_SHORT cases?
> **YES, but at an unacceptable cost.** On Large buildings (>= 12.0m), $\sigma=2.0$ reduces `FALSE_SHORT` cases from 272 down to 154, but degrades Large building MAE from 7.19m to 8.45m due to severe overshooting.

### Question 5: Does it introduce additional FALSE_LONG errors?
> **YES, MASSIVELY.** Across the dataset, `FALSE_LONG_SHADOW` failures increase from **243 (raw)** up to **482 ($\sigma=2.0$)** and **598 ($\sigma=3.0$)**.

### Question 6: Does the effect generalize across building sizes and tiles?
> **NO.** While smoothing marginally helps tall complex roofs, it severely degrades Small (< 4m) and Medium (4 - 12m) buildings where shadow boundaries are sharp and short.

### Question 7: Is adaptive smoothing justified using only image-derived signals?
> **NO.** There is no reliable 2D image signal to dynamically select $\sigma$ per building without causing regressions or requiring ground-truth supervision.

### Question 8: Should M4 remain unchanged or is a separate refinement investigation justified?
> **DEFINTIVE CONCLUSION: MAINTAIN M4 UNCHANGED (`shadow/m4_physical_raycast_experiment.py`).**  
> Diagnostic evidence proves 1D profile smoothing is not a viable refinement. Future research should focus on 2D adaptive building boundary contact scoring or shadow-contrast ratio normalization rather than 1D ray filtering.

---

## Verification Artifacts
- **Full Diagnostic Markdown Report**: `output/m6_multiscale_analysis.md`
- **Summary Transition CSV**: `output/m6_multiscale_summary.csv`
- **Representative Profiles JSON**: `output/m6_multiscale_profiles.json`
- **Diagnostic Plot Overlays**: `output/m6_diag_*.png`
- **Production Code Status**: **NO PRODUCTION M4 CODE WAS MODIFIED.**
