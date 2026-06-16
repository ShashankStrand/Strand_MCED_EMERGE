# Odds Ratio and Inverse Probability Weighting Analysis Pipeline

This script performs confounder analysis and odds ratio estimation for cancer detection model scores using logistic regression and inverse probability weighting (IPW).

It supports:

* Unadjusted odds ratio (OR) estimation
* Multivariable-adjusted OR estimation
* Weighted OR using stabilized inverse probability weights
* Standardized Mean Difference (SMD) balance checks

The pipeline is designed for confounder sensitivity analysis.

---

# What This Script Computes

For each model score column (split):

1. Unadjusted logistic regression

* Outcome ~ Score
* Odds Ratio (OR)
* 95% CI
* p-value
* AUC

2. Multivariable-adjusted logistic regression

* Outcome ~ Score + covariates
* Adjusted OR
* p-value
* AUC

3. Inverse Probability Weighting (IPW)
   For each covariate treated as exposure:

* Fit propensity score model using other covariates
* Compute stabilized weights
* Fit weighted GLM
* Estimate weighted OR and CI
* Compare with unweighted OR

4. Covariate Balance Diagnostics

* Standardized Mean Difference (SMD)
* Before weighting
* After weighting
* For all non-exposed covariates

---

# Input File

The script expects:

../InputFiles/Cohort1.tsv

Required columns:

Cancer_type
Age
Gender
Tobacco
Plasma Storage Time
Flow_cell
Collection Date
Sequencing Date
Extraction Date

Plus score columns for each model split, such as:

RandomSplit_Score
SeqSplit_Score
CollDate_Score
Site_Score
AltSeqSplit_Score
AltCollDate_Score
AltSite_Score

---

# Preprocessing Performed

The script automatically:

* Parses date columns
* Binarizes covariates:

Gender → Male = 1 else 0
Tobacco → Habit = 1 else 0
Outcome → cancer = 1, control = 0
FC → flowcell contains “B” = 1 else 0

Derived binary covariates:

Bin_Age → Age above mean
Bin_Plasma Storage Time → above mean

---

# Statistical Methods Used

Logistic Regression:

* statsmodels GLM (Binomial family)
* Reports coefficients, OR, CI, p-values
* AUC computed from predicted probabilities

Odds Ratio:

* OR = exp(beta)
* CI = exp(confidence interval of beta)

Inverse Probability Weighting:

* Propensity score via logistic regression
* Stabilized weights:
  exposed: p / ps
  unexposed: (1-p)/(1-ps)

Weighted GLM:

* Uses freq_weights argument

Balance Check:

* Standardized Mean Difference (SMD)
* Weighted vs unweighted comparison

Nonparametric Testing:

* Mann–Whitney U test for subgroup score comparison

---

# Key Functions

compute_smd()

* Computes standardized mean difference
* Supports weighted and unweighted calculation

LR()

* Fits logistic regression model
* Returns OR table with CI, p-values, AUC

computePVal()

* Mann–Whitney U test per subgroup

binarize()

* Maps continuous covariates to binary versions

---

# Model Score Columns Used

Score columns are mapped to split indicators:

splitCols = {
Site_Score → Site_Split
CollDate_Score → CollDate_Split
SeqSplit_Score → Seq_Split
RandomSplit_Score → Random_Split
AltSeqSplit_Score → AltSeq_Split
AltCollDate_Score → AltCollDate_Split
AltSite_Score → AltSite_Split
}

Each score is evaluated independently.

---

# Outputs

## Logistic Regression Summary

confounder_summary_LR.tsv

Contains:

* Covariate
* OR
* OR CI
* p-value
* AUC
* Model covariate list

Includes:

* Unadjusted models
* Adjusted models

---

## Weighted OR + Balance Metrics

All_balanceORs.tsv

Contains:

* Score column
* Exposure covariate
* Weighted OR
* Unweighted OR
* CI (parametric)
* p-values
* SMD before weighting
* SMD after weighting

Used to verify covariate balance improvement after IPW.

---

# Dependencies

Install required packages:

pip install numpy pandas scipy scikit-learn statsmodels seaborn matplotlib

---

# Execution

Run from script directory:

python confounder_or_ipw.py

Ensure path exists:

../InputFiles/Cohort1.tsv

---
