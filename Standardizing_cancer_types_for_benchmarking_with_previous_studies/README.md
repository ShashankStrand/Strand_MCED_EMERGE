# Stage-Wise Standardized Sensitivity Analysis with Bootstrap CIs

This script performs stage-wise standardized sensitivity estimation across multiple cancer detection tests using a binomial GLM and bootstrap resampling.

It supports:

* Detection rate modeling using GLM (binomial)
* Standardized sensitivity estimation by clinical stage
* Stage-wise sensitivity normalization across cancer types
* Bootstrap confidence intervals for standardized sensitivity
* Bootstrap hypothesis testing between tests
* Sensitivity difference estimation between tests (stage-wise)
* P-value estimation from bootstrap distributions

The script is designed for comparative performance analysis between cancer detection assays.

---

# Statistical Framework

The script fits a binomial GLM:

prop ~ test * stage + cancer_type

Where:

prop = positive / total
test = assay name
stage = clinical stage (I–IV)
cancer_type = cancer category

Weights:

* freq_weights = total sample count per row
* This ensures modeling detection rate using aggregated counts

This model estimates detection probability while adjusting for:

* Stage distribution
* Cancer type mix
* Test-stage interaction effects

---

# Core Outputs

For each cancer-type subset, the script produces:

1. Stage-wise standardized sensitivity per test
2. Bootstrap confidence intervals per stage
3. Bootstrap differences between tests
4. Bootstrap p-values for superiority testing

---

# Input File

Reads:

../InputFiles/Comparison-Klein.tsv

Expected columns:

Study
Cancer Class
Clinical Stage
Total
Test Positive

Columns are renamed internally to:

Study → test
Cancer Class → cancer_type
Clinical Stage → stage
Total → total
Test Positive → positive

Derived columns:

failures = total − positive
prop = smoothed positive rate

---

# Cancer Subsets Evaluated

The script runs analysis separately for predefined cancer groups:

All14
CCGAPreSpec9
nonScreenable9
cSEEK8

Each subset filters cancer_type before modeling.

---

# Key Functions

## linearmodel(df, w)

Fits binomial GLM:

prop ~ test * stage + cancer_type

Arguments:

* df = input dataframe
* w = weight column (typically "total")

Returns:

* Fitted statsmodels GLM result

---

## standardized_sensitivity(result, df, test_value)

Computes standardized stage-wise sensitivity.

Process:

* Builds prediction dataframe using:
  stage, cancer_type, total
* Sets test column = selected test_value
* Uses GLM to predict detection probability
* Aggregates by stage using weighted mean
* Weights = total samples

Returns:
DataFrame with:

stage
Sensitivity
test

This produces stage-normalized sensitivity independent of cancer mix.

---

## bootstrap_ci(df, n_boot)

Computes bootstrap confidence intervals for standardized sensitivity.

Process:

* Resample rows with replacement
* Refit GLM on each bootstrap sample
* Compute standardized sensitivity
* Store results
* Compute 2.5% and 97.5% quantiles

Returns:

test
stage
CI_low
CI_high

---

## bootstrap_stage_diffs_multitest(df, ref_test, cur_test)

Computes bootstrap distribution of sensitivity differences.

For each bootstrap iteration:

* Resample dataset
* Fit GLM
* Compute standardized sensitivity for both tests
* Compute stage-wise difference:

cur_test − ref_test

Returns:
Matrix of bootstrap differences:

rows = bootstrap iterations
columns = stages

---

# Bootstrap Hypothesis Testing

For each stage:

diff = cur − reference

Statistics reported:

mean difference
2.5% percentile
97.5% percentile
p-value = fraction(diff ≤ 0)

This corresponds to:

H0: current ≤ reference
H1: current > reference

One-sided superiority test.

---

# Model Smoothing

To avoid numerical instability:

prop = (positive + 5e-8) / (total + 1e-4)

Prevents zero or one proportions causing GLM convergence issues.

---

# Stage Handling

Stages are ordered categorical:

I, II, III, IV

This ensures correct model encoding and ordered output.

---

# Outputs Written

For each cancer subset (k):

## Standardized Sensitivity CIs

../Tables/{k} CIs.tsv

Contains:

test
stage
Sensitivity
CI_low
CI_high

---

## Bootstrap Test Comparison vs CCGA

../Tables/{k} CCGA-pvalue.tsv

Contains per stage:

diff
lcl
ucl
p

---

## Bootstrap Test Comparison vs CancerSEEK

../Tables/{k} CancerSEEK-pvalue.tsv

Same structure as above.

---

# Dependencies

Required packages:

numpy
pandas
scipy
statsmodels
matplotlib

Install:

pip install numpy pandas scipy statsmodels matplotlib

---

# Execution

Run from script directory:

python standardized_sensitivity_bootstrap.py

Ensure directory exists:

../InputFiles/Comparison-Klein.tsv
../Tables/

---

# Methodological Notes

* Uses binomial GLM with frequency weights
* Standardization removes cancer-type composition bias
* Bootstrap resampling is row-level
* Sensitivity aggregation is weighted by total samples
* Confidence intervals are percentile bootstrap
* P-values are empirical bootstrap probabilities

---
