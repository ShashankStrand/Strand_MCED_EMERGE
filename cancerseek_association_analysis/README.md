Logistic regression analysis of the **CancerSEEK (Cohen)** dataset, testing the association between plasma DNA concentration and cancer detection, adjusting for tumor type and AJCC stage. Also produces stage-wise, yield-tertile sensitivity estimates.

## Input

Supplementary Table 7 — Cohen/CancerSEEK tab (TSV):
- `../InputFiles/CancerSeek-S5.tsv`

## Output

- Association statistics printed to the console: GLM (logistic regression) summary and Wald tests per term.
- A stage-wise tertile sensitivity table written to `../Tables/CancerSeek_Stage_wise_Tertile_Sens.tsv`.

## What the script does

1. **Load and clean** the CancerSEEK data; standardize column names; convert yield/mutant columns to numeric.
2. **Filter** to AJCC stage I–IV samples (sensitivity loop uses I–III).
3. **Define the outcome:** `detected = 1` if the CancerSEEK test result is "Positive".
4. **Fit a logistic regression** of detection against tumor type, stage, and plasma DNA concentration (with a spline term).
5. **Stratified sensitivity:** for each stage, splits samples into yield tertiles and computes sensitivity (with 95% CIs), then combines into one table.

## Key columns

- `CancerSEEK_Test_Result` — detection outcome.
- `cfDNA_Yield` (from `Plasma_DNA_concentration_(ng_per_mL)`) — plasma DNA concentration.
- `Tumor_type` — tumor type.
- `AJCC_Stage` — AJCC stage.

There is an optional filter (commented out) to restrict to samples where the mutation is present in the tumor (`In_Tumor == "Present"`).

## Requirements

Python with: `pandas`, `numpy`, `scipy`, `statsmodels`, `seaborn`, `matplotlib`.

## Usage

```bash
python cancerseek_association_analysis.py
```

Run from a directory where `../InputFiles/` holds the input TSV and `../Tables/` exists for output.
