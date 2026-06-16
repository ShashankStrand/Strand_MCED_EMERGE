# Confidence Interval & ROC Evaluation Pipeline

This repository contains a Python evaluation pipeline for computing classification performance metrics for cancer detection models, including:

* Sensitivity / Specificity
* Confusion matrix counts (TP, TN, FP, FN)
* ROC curves
* AUC
* Bootstrap/Wilson confidence intervals
* Subgroup analysis by:

  * Stage
  * Cancer type
  * Cohort split
* ROC plots with bootstrap confidence bands

---

# Overview of What This Script Does

For each cohort split file:

* Converts labels into binary truth (control = 0, cancer = 1)
* Applies score threshold derived from desired training specificity
* Computes All of the above metrics.

* Performs subgroup analysis.
* Saves:

  * Final metrics table (TSV)
  * ROC curve plots with CI band (PNG)

---

# Input File Requirements

Input files must be **TSV** and contain at least these columns:

| Column     | Description                                            |
| ---------- | ------------------------------------------------------ |
| `sampleId` | Sample ID column (e.g. Run0248_ED-27-054C-ABK)         |
| `type`     | Label column (must include `control` and cancer types) |
| `stage`    | Cancer stage (e.g. I, II, III, IV, Benign)             |
| `gender`   | Gender columns (e.g. Male, Female)                     |
| `GenCncrs` | Model prediction score (probability or risk score)     |

Files are expected in:

```
../InputFiles/
```

Examples:

```
RandomCV-80.tsv
RandomTest.tsv
SiteCV-80.tsv
SeqTest.tsv
```

---

# Output Files

## Metrics Table

Saved to:

```
../Tables/Confidence Intervals-<specificity>.tsv
```

## ROC Figures

Saved to:

```
../SuppFigs/
```

Each ROC plot contains:

* ROC curve
* Bootstrap CI band
* AUC with CI annotation

---

# Dependencies

Install required packages:

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib
```

---

# Key Parameters (Top of Script)

You can adapt these to your dataset:

```python
LABEL_COL = "type"
CONTROL_LABEL = "control"
SCORE_COL = "GenCncrs"
STAGE_COL = "stage"
TYPE_COL = "type"

N_BOOTSTRAPS = 1000
CI_PCT = 95
```

---

# Threshold Selection Logic

Thresholds are automatically computed per experiment using control samples:

```python
cutoffs[e] = df[df[TYPE_COL]=='control']['score'].quantile(scutoff)
```

Example:

```
scutoff = 0.90   → 90% specificity threshold
scutoff = 0.99   → 99% specificity threshold
```

This threshold is then used consistently across:

* Test sets
* Rare cohorts
* Benign cohorts

---

# Methods Used

## Sensitivity / Specificity

Computed from confusion matrix at fixed threshold.

## Confidence Intervals

Two CI methods are implemented:

### Bootstrap CI

Used for:

* Overall metrics
* Stage-wise metrics

Procedure:

* Resample dataset with replacement
* Recompute metrics
* Take percentile CI

### Binomial Proportion CI (statsmodels)

Used for:

* Cancer-type subgroup metrics

Supported methods:

* Wilson (default)
* Beta (Clopper–Pearson)
* Jeffreys
* Agresti–Coull

---

# ROC & AUC CI

ROC AUC CI is computed via bootstrap:

* Resample rows
* Recompute AUC
* Percentile CI
* Interpolate TPR curves
* Build CI band across FPR grid

---

# Main Execution Flow

The script runs:

```
calculate() →
    Overall metrics
    Stage-wise metrics
    Cancer-wise metrics
    Cancer+Stage metrics
```

Then concatenates results and writes:

```
Confidence Intervals-<spec>.tsv
```

---

# Running the Script

From project directory:

```bash
python3 ComputeConfidenceIntervals.py
```

Ensure folders exist:

```
../InputFiles/
../Tables/
../SuppFigs/
```

---
