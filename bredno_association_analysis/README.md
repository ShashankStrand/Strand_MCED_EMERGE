Logistic regression analysis of the **Bredno (WGBS)** dataset, testing the association between cfDNA concentration and cancer detection, adjusting for tumor type and AJCC stage. Also produces stage-wise, yield-tertile sensitivity estimates.

## Input

Supplementary Table 7 — Bredno tab (TSV):
- `../InputFiles/Bredno.tsv`

## Output

- Association statistics printed to the console: GLM (logistic regression) summary and Wald tests per term.
- A stage-wise tertile sensitivity table written to `../Tables/Bredno_Stage_wise_Tertile_Sens.tsv`.

## What the script does

1. **Load and clean** the Bredno data; standardize column names; convert cfDNA yield to numeric.
2. **Filter** to samples with valid yield and AJCC stage I–IV.
3. **Define the outcome:** `detected = 1` if the WGBS classifier result is "detected".
4. **Fit a logistic regression** of detection against tumor type (`Source`), stage, and cfDNA yield (with a spline term).
5. **Stratified sensitivity:** for each stage, splits samples into yield tertiles and computes sensitivity (with 95% CIs), then combines into one table.

## Key columns

- `wgbs_classifier_result` — detection outcome.
- `cfDNA_Yield` (from `cfdna_conc_ng_ml`) — cfDNA concentration.
- `Source` — tumor type.
- `clinical_stage` — AJCC stage.

## Requirements

Python with: `pandas`, `numpy`, `scipy`, `statsmodels`, `seaborn`, `matplotlib`.

## Usage

```bash
python bredno_association_analysis.py
```

Run from a directory where `../InputFiles/` holds the input TSV and `../Tables/` exists for output.
