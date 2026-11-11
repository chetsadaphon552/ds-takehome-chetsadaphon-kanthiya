# Data Scientist Take-home Assessment

**Candidate:** Chetsadaphon Kanthiya  
**Problems submitted:** 1 (Discrete Mathematics), 2 (Linear Algebra), and 4 (Machine Learning)

## Repository map

```text
README.md
AI_USAGE.md
requirements.txt
report/
  q1/                  # Problem 1 proofs and benchmark figures
  q2/                  # Problem 2 proofs and PCA figure
  q4/                  # Generated Q4 figures
src/
  q1/                  # SCC algorithm, blocking analysis, and tests
  q2/                  # Ownership/PCA notebook and PCA analysis
  q4/
    output/
      data/             # Training/holdout feature and target CSVs
      models/           # Trained model and scaler pickle files
      results/          # CV, OOF, threshold, and holdout result files
```

`src/` contains code and notebooks. Q4 generated data, models, and results are
organized under `src/q4/output/`; figures intended for the report are saved to
`report/q4/`. The original UCI spreadsheet is in `src/q4/data/` and is below
the 50 MB limit.

## Setup

Use Python with the pinned packages in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the included dataset needs to be restored, run this from the repository root:

```powershell
python src\q4\0_download_data.py
```

## Reproduce the work

### Problem 1

Run these commands from the repository root:

```powershell
python src\q1\1_circular_shareholding.py
python src\q1\2_benchmark_comparison.py
python src\q1\3_entity_resolution.py
python src\q1\5_test_graph.py
```

- The SCC algorithm and its complexity are in `src/q1/1_circular_shareholding.py`.
- The 100,000-node benchmark is in `src/q1/2_benchmark_comparison.py`.
- The blocking comparison calculation is in `src/q1/3_entity_resolution.py`.
- The topological-order proof is `report/q1/mathematical_proof.txt`.
- The blocking-count convexity proof is `report/q1/convexity_proof.txt`.

### Problem 2

Open and run `src/q2/1_ownership_pca.ipynb`. It contains the hand derivation,
NumPy ownership calculation, PCA proofs, SVD discussion, and standardization
example. Then run the PCA tuning analysis from the repository root:

```powershell
python src\q2\2_pca_tuning_analysis.py
```

The Neumann-series proof is `report/q2/convergence_proof.txt`; the PCA
Lagrange-multiplier proof is `report/q2/lagrange_proof.txt`.

### Problem 4

The Q4 notebooks must be run in order, from a Jupyter server whose working
directory is `src/q4`:

1. `1_eda_final.ipynb` — data quality, EDA, feature engineering, and the
   stratified 70/30 split.
2. `2_modeling_final.ipynb` — compares engineered-feature sets using the same
   training-only stratified 5-fold CV and logistic-regression baseline. It
   compares core features, each engineered feature added alone, all six
   together, and leave-one-out sets. It applies the one-standard-error rule
   to prefer the smallest set close to the top ROC-AUC, saves
   `feature_set_comparison.csv` and the selected feature list, then uses that
   same set for model tuning, comparison, and OOF predictions. Models go to
   `output/models/`; CSV/JSON results go to `output/results/`.
3. `3_evaluation_final.ipynb` — selects the 5:1 cost threshold from training
   out-of-fold predictions, verifies the saved holdout, and evaluates the
   selected model on that holdout. It also reports calibration, a pre-specified
   paired bootstrap CI for LightGBM versus logistic-regression ROC-AUC, SHAP,
   and group metrics. This run checks calibration with the Brier score and
   reliability curve but does not fit a post-hoc calibrator; `SEX`, `AGE`, and
   `EDUCATION` remain model inputs.

For example:

```powershell
Set-Location src\q4
jupyter lab
```

The notebooks write report figures under `../../report/q4/`. Generated files
are stored beneath `src/q4/output/` in three folders: `data/` for the
train/holdout CSVs, `models/` for pickle files, and `results/` for CV/OOF and
final evaluation outputs. The notebook paths are relative to the required
`src/q4` working directory. Re-run all three notebooks in order before using
the Q4 figures or metrics in a submission.

## Documentation and generated files

| File/folder | Purpose and whether it is used |
|---|---|
| `README.md` | Main index, selected problems, setup, and reproduction instructions; use this to rerun the work. |
| `AI_USAGE.md` | Required disclosure of AI tools and verification; include it in the submission. |
| `report/q1/*.txt`, `report/q2/*.txt` | Written proofs corresponding to the specified assessment parts; no script reads these files. |
| `src/q4/output/data/training_features.csv`, `training_target.csv` | Written by notebook 1 for reproducibility; notebook 2 recreates the same stratified split using the fixed seed. |
| `src/q4/output/data/holdout_features.csv`, `holdout_target.csv` | Holdout written by notebook 1 and verified by notebook 3. |
| `src/q4/output/models/*.pkl` | Trained logistic regression, tree/boosting candidates, and feature scaler written by notebook 2. |
| `src/q4/output/results/feature_set_comparison.csv` | Same-fold CV comparison: core only, each engineered feature added separately, all six added, and each leave-one-out set. Includes ROC-AUC/PR-AUC means, standard deviations, deltas against core/full sets, and flags for sets within one standard error and the selected set. |
| `src/q4/output/results/selected_features.json` | Feature set selected by the one-standard-error rule using training CV; notebook 3 uses these exact columns for holdout evaluation. |
| `src/q4/output/results/training_oof_predictions.csv` | Training-only out-of-fold probabilities from the selected feature set; used by notebook 3 to choose the cost threshold without holdout leakage. |
| `src/q4/output/results/cross_validation_model_comparison.csv` | Cross-validation model metrics used by notebook 3 to select the model. |
| `src/q4/output/results/training_bootstrap_confidence_intervals.csv`, `model_feature_importance.csv` | Supporting model-analysis outputs from notebook 2. |
| `src/q4/output/results/final_holdout_evaluation.csv`, `selected_decision_threshold.json` | Final holdout metrics and selected threshold written by notebook 3 for reporting. |
| `report/q1/`, `report/q2/`, `report/q4/` figures | Generated visual results; these are not inputs to the code and should be refreshed by rerunning the respective analyses. |

## Results and submission checks

The preprocessing step creates a 19-feature candidate pool (13 core inputs,
including `SEX`, plus six engineered features). Notebook 2 compares subsets
from this pool using training-only CV; the fitted models and final holdout
evaluation use the selected 14-feature subset: the 13 core features plus
`NUM_DELAYS`. The selected model inputs are `PAY_0`, `PAY_2`, `PAY_3`,
`PAY_4`, `PAY_5`, `PAY_6`, `LIMIT_BAL`, `BILL_AMT1`, `BILL_AMT6`,
`PAY_AMT1`, `AGE`, `EDUCATION`, `SEX`, and `NUM_DELAYS`. In the latest Q4
run, the best observed feature-set ROC-AUC was 0.7549 and the selected set
scored 0.7548 in training cross-validation.
LightGBM was the top model in training
cross-validation, with holdout ROC-AUC 0.7747 and PR-AUC 0.5517 at the
OOF-selected threshold of 0.385. Holdout cost was 5,093 versus 5,260 at the
fixed 0.50 threshold under the stated 5:1 cost ratio. These results are in
`src/q4/output/results/`; they do not prove the selected features are
universally optimal or establish real-world performance. Run notebooks 1
through 3 in order to reproduce them. The assessment report PDF is submitted
separately and is not included in this repository; refresh it if a later run
changes the outputs.
The reported subgroup metrics are descriptive rather than a fairness
certification. Before any pilot, validate on later-period data, assess legal
and fairness risks, establish human oversight, and define drift/rollback
monitoring (including PSI) against a pre-deployment benchmark.

See `AI_USAGE.md` for the AI-use disclosure. Before submission, confirm that the
repository has the required Git history and that the final report contains the
latest executed results.
