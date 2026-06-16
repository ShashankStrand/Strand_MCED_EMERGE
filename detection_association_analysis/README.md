Logistic regression analysis to test which clinical and technical variables are associated with cancer detection in a cfDNA assay.

## Input

Supplementary Table 1 (TSV files):
- `../InputFiles/cohort-main.tsv` — main cohort tab
- `../InputFiles/Benign.tsv` — benign cohort tab

The two cohorts are combined. The main cohort is restricted to the Independent Validation Set (`Random_Split == "LeaveOut"`).

## Output

Association statistics printed to the console:
- Sample counts per cancer type
- Number of rows dropped due to missing values
- Full GLM (logistic regression) summary with coefficients and p-values
- Wald test results per model term
- Bootstrapped AUC with 95% confidence interval

## What the script does

1. **Load and combine** the main and benign cohorts, keeping a fixed set of columns.
2. **Clean and transform** variables:
   - Bins collection time into 3-hour windows.
   - Maps flow cells to sequencer types (NovaSeq X+ / NovaSeq 6000).
   - Groups rare cancer types (n < 3) into "Other".
3. **Define the outcome:** a sample is "detected" if its score exceeds the 99% specificity threshold (`0.9906`) set during training.
4. **Fit a logistic regression** of detection against cancer type, stage, cfDNA yield (with a spline term), age, sex, tobacco, alcohol, flow cell, plasma storage time, BMI, DNA input, and coverage.
5. **Evaluate** the model using the GLM summary, Wald tests, and a bootstrapped AUC.

## Key settings

- `threshold = 0.9906` — detection cutoff (99% specificity in training).
- `yield_col` — set to cfDNA yield per mL of plasma (`tape_col`); can be switched to the Qubit yield (`qubit_col`).
- The model formula can be swapped for an alternate version (commented in the script) that includes a stage × yield interaction.

## Helper functions

- `auc_ci_bootstrap()` — computes a bootstrapped confidence interval for the AUC.
- `forest()` — generates a forest plot of odds ratios (currently not called; uncomment `forest(model)` to use).

## Requirements

Python with: `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `patsy`, `matplotlib`.

## Usage

```bash
python detection_association_analysis.py
```

Run from a directory where `../InputFiles/` contains the two input TSV files.
