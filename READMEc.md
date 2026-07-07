# SR-CCP: Structural Regularization Cost-Complexity Pruning

**Improving Tree Structure with SR-CCP: A Regularized Pruning Strategy for Shallower and Simpler Decision Models**

A novel decision tree pruning method submitted to the **International Journal of Computational Intelligence Systems and Data Science (IJCDS)**. This project contains the implementation, experiments, and analysis addressing reviewer feedback for the paper.

## Overview

Standard CCP selects a subtree by minimizing `R(T) + α|T|` (error + complexity). SR-CCP reformulates this as:

```
R(T) + α'·(1 + |R_train - R_val|) + λd·depth(T) + λf·(n_features_used / n_total_features)
```

Where:
- **α'·(1 + |R_train - R_val|)** adjusts the cost-complexity parameter based on the generalization gap
- **λd·depth(T)** penalizes tree depth, encouraging shallower trees
- **λf·(feat_ratio)** penalizes feature usage, encouraging simpler models

Hyperparameters `λd` (depth_penalty) and `λf` (feature_penalty_weight) are tuned per dataset via **Optuna** Bayesian optimization.

## Compared Pruning Methods

| Method | Description |
|--------|-------------|
| **Base** | Unpruned decision tree (baseline) |
| **CCP** | Standard cost-complexity pruning |
| **SR-CCP** | Proposed method with depth + feature regularization |
| **REP** | Reduced Error Pruning |
| **MEP** | Minimum Error Pruning |
| **DBP** | Depth-constrained pruning (max_depth=5) |
| **XGBoost** | Gradient-boosted trees with Optuna-tuned pruning (gamma, reg_alpha, reg_lambda) + early stopping (Obs 12) |

## Datasets (7)

| Dataset | Domain | Samples | Features |
|---------|--------|---------|----------|
| BanknoteAuth | Security | 1,372 | 4 |
| BreastCancer | Medical | 569 | 30 |
| HeartDisease | Medical | 303 | 13 |
| Ionosphere | Radar | 351 | 34 |
| KC2 | Software defects | 522 | 21 |
| QSARBiodeg | Chemistry | 1,055 | 41 |
| SpamBase | Email | 4,601 | 57 |

## Key Findings

- **Dramatically simpler trees** — e.g., SpamBase: 477.8 nodes (Base) → 141.8 (CCP) → 11.8 (SR-CCP); depth from 29 to 4.6
- **Competitive accuracy** — SR-CCP matches/exceeds CCP accuracy on most datasets while using 80-95% fewer nodes
- **Ablation** — Both depth and feature penalties contribute independently; full SR-CCP yields the best structural simplification
- **Statistical significance** — Confirmed via Friedman + Nemenyi CD tests, Wilcoxon signed-rank (Benjamini-Hochberg/FDR), and paired t-test (Benjamini-Hochberg/FDR)

## Reviewer Observations & Implementation Status

The journal reviewers provided 24 observations. Below is the status of each:

| # | Observation | Status | Notes |
|---|-------------|--------|-------|
| 1 | Proofreading for grammar, punctuation, and academic writing | Pending | |
| 2 | Revise abstract to separate problem, method, strategy, setup, findings | Pending | |
| 3 | Condense introduction, emphasize CCP limitations | Pending | |
| 4 | Explicitly state the research gap (why existing pruning fails to jointly optimize accuracy, depth, features) | Pending | |
| 5 | Emphasize novelty by explicitly distinguishing SR-CCP from CCP, REP, MEP, DBP | Pending | |
| 6 | Strengthen related work with recent studies (2024-2026) | Pending | |
| 7 | Clearer step-by-step methodology (tree generation, pruning path, Bayesian optimization, regularization, selection) | Pending | |
| 8 | Expand mathematical formulation with clearer explanation of objective function and each regularization parameter | Pending | |
| 9 | Detail hyperparameter optimization (Bayesian settings, search iterations, convergence, cost) | Pending | |
| 10 | Complete implementation details (libraries, hardware, seeds, reproducibility) | Pending | |
| **11** | **Statistical significance analysis (Friedman/Nemenyi, Wilcoxon, t-test)** | **Implemented** | Friedman, Nemenyi CD, Wilcoxon (Benjamini-Hochberg/FDR), paired t-test (Benjamini-Hochberg/FDR) for accuracy & structural metrics. See Cell 7. |
| **12** | **Additional comparisons with recent interpretable tree algorithms** | **Partially Implemented** | XGBClassifier (Optuna-tuned gamma/reg_alpha/reg_lambda + early stopping) added, see Cell 3 (`train_xgboost`). HSTreeClassifier (imodels) still pending |
| 13 | Deeper discussion of trade-offs (performance vs simplicity vs features vs inference) | Pending | |
| **14** | **Ablation study for depth penalty and feature penalty components** | **Implemented** | 4 configurations: Full SR-CCP, No Depth, No Feature, No Penalties. See Cell 4 & `ablation_study/`. |
| 15 | Computational complexity analysis and scalability discussion | Pending | |
| 16 | Strengthen conclusion (contributions, limitations, future work) | Pending | |
| 17 | Figures cited and discussed before they appear | Pending | |
| 18 | High-resolution figures (≥300 DPI or SVG/PDF) | Pending | |
| 19 | Workflow diagram (data preparation → tree generation → Bayesian optimization → SR-CCP → evaluation) | Pending | |
| 20 | Statistical comparison chart with significance indicators | Pending | |
| **21** | **Parameter sensitivity analysis (λd and λf)** | **Implemented** | Vary λd [1e-4, 1e-2] and λf [0.01, 0.2] independently. See Cell 10 & `sensitivity_analysis/`. |
| **22** | **Computational efficiency comparison (training, pruning, inference time + memory)** | **Implemented** | Train_Time_ms, Prune_Time_ms, Inference_Time_ms, Memory_KB, Peak_Memory_KB. See Cell 3 & FINAL_RESULTADOS.csv. |
| 23 | Tables follow journal formatting (captions above, consistent decimal precision) | Pending | |
| 24 | References cited in order of appearance | Pending | |

### Code-Related Observations Detail

See `OBSERVACIONES_CODIGO.md` for full implementation notes on the 5 code-related observations:

| Obs | Description | Difficulty | Est. Time | New Dependencies |
|-----|-------------|------------|-----------|------------------|
| 22 | Computational efficiency | Low | 30 min | None |
| 11 | Statistical significance | Low-Medium | 45 min | None |
| 14 | Ablation study | Medium | 1 hour | None |
| 21 | Parameter sensitivity | Medium | 1.5 hours | None |
| 12 | Additional methods (HSTree, XGBoost) | Medium-High | 2 hours | imodels, xgboost — XGBoost done, HSTree pending |

## Repository Structure

```
├── SR_CPP_CODE.ipynb              # Main notebook with all experiments
├── FINAL_RESULTADOS.csv           # Full results (accuracy, structure, efficiency)
├── FINAL_RESULTADOS_structural_significance.csv
├── OBSERVACIONES_CODIGO.md        # Code implementation notes for reviewer observations
├── paper/
│   ├── RESULTADOS_PAPER.csv       # Paper-reported results
│   ├── IJCDS_Observaciones.csv    # Full reviewer feedback (24 observations)
│   └── Improving_Tree_Structure... # Paper PDF
├── ablation_study/
│   ├── ablation_results.csv
│   ├── ablation_accuracy.png
│   ├── ablation_complexity.png
│   └── ablation_plots.ipynb
├── sensitivity_analysis/
│   ├── obs21_sensitivity_results.csv
│   ├── obs21_depth_penalty_sensitivity.png
│   └── obs21_feature_penalty_sensitivity.png
├── Bugs/                          # Bug fix notes
├── paper_Accuracy_nemenyi_cd_diagram.png
└── nemenyi_cd_diagram.png
```

## Requirements

```
numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, optuna, joblib, jupyter, xgboost
```

## Usage

Open `SR_CPP_CODE.ipynb` in Jupyter:

1. **Cell 1-2**: Imports and dataset loading
2. **Cell 3**: Training functions for all 7 methods (incl. XGBoost, Obs 12) + evaluation + computational efficiency (Obs 22)
3. **Cell 4**: Ablation study — 4 SR-CCP configurations (Obs 14)
4. **Cell 5**: n_trials ablation (Optuna budget sensitivity)
5. **Cell 6**: Results aggregation + CSV export
6. **Cell 7**: Statistical significance — Friedman, Nemenyi CD, Wilcoxon, t-test (Obs 11)
7. **Cell 10**: Parameter sensitivity — vary λd and λf (Obs 21)

## Results

- Paper-reported results: `paper/RESULTADOS_PAPER.csv`
- Full experimental results: `FINAL_RESULTADOS.csv`
- Ablation results: `ablation_study/ablation_results.csv`
- Sensitivity analysis: `sensitivity_analysis/obs21_sensitivity_results.csv`
- Structural significance: `FINAL_RESULTADOS_structural_significance.csv`

## Citation

If you use this work, please cite the associated paper (see `paper/` directory).
