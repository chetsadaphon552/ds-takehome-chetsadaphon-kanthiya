# AI Usage Disclosure

AI tools were used during this take-home assessment. The candidate is
responsible for reviewing, understanding, and being able to explain every
submitted result and code change.

## Tools used

- **Kiro (Claude Sonnet / Anthropic):** Interactive coding support during the
  assessment, including implementation and report drafting.
- **Copilot SDK in VS Code:** Repository review, code-comment and documentation
  edits, Q4 output-path cleanup, and correction of the Q4 split, threshold,
  evaluation, and artifact paths.

## Scope and verification

| Area | AI assistance | Verification performed |
|---|---|---|
| Problem 1 | Proof structure, iterative strongly connected component implementation, and blocking calculations | The graph tests were run (4 passed). The blocking helper was checked for divisible and non-divisible block sizes; fabricated recall estimates were removed because no labelled duplicate pairs are available. |
| Problem 2 | Explanations and code scaffolding for indirect ownership, PCA, and SVD | Ownership values and calculations were compared with the matrix in the notebook; proof documents were checked for notation, calculations, convergence conditions, and careful distinction between indirect ownership and voting control. |
| Problem 4 | Feature engineering, model-comparison/evaluation workflow, output organization, report drafting, and subgroup-result wording | The final Q4 workflow was run on 2026-11-12. The saved train/holdout split, 14 selected features, model comparison, holdout metrics, and displayed evaluation outputs were checked for consistency. The final model retains AGE and SEX as inputs and does not apply post-hoc calibration. |
| Repository documentation | README, AI disclosure, proof documents, and submission-artifact inventory | Paths and current file references were checked against the repository. |

## Important limitation

The current Q4 artifacts use the 19-feature candidate pool and the
14-feature set selected by training-only cross-validation. The holdout
evaluation reports uncalibrated model probabilities and descriptive subgroup
metrics; neither should be interpreted as a fairness certification or
production validation. Review the results and refresh the report PDF if a later
run changes the saved metrics.

## Data source

The Q4 analysis uses the public UCI Default of Credit Card Clients dataset
included in `src/q4/data/`. No personal or organizational internal data was
used.
