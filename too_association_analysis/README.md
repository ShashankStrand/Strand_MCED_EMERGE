Logistic regression analysis to test which clinical and technical variables are associated with correct **Tissue-of-Origin (TOO)** prediction in a cfDNA assay.

## Input

Supplementary Table 1 (TSV files):
- `../InputFiles/cohort-main.tsv` — main cohort tab
- `../InputFiles/Benign.tsv` — benign cohort tab

Supplementary Table 5 (TOO prediction results):
- `../InputFiles/TOO-Ge80Lt20FragTop300-CV.tsv` — cross-validation set
- `../InputFiles/TOO-Ge80Lt20FragTop300-Test.tsv` — test set
- `../InputFiles/TOO-Ge80Lt20FragTop300-Benign.tsv` — benign set

The cohort tables and TOO results are merged on sample ID. The analysis keeps only the benign and independent validation samples (`Random_Split != "LeaveIn"`).

## Output

Association statistics printed to the console:
- Sample count after filtering
- Full GLM (logistic regression) summary with coefficients and p-values
- Wald test results per model term

## What the script does

1. **Load and combine** the main + benign cohorts with the TOO prediction results.
2. **Clean and transform** variables:
   - Bins collection time into 3-hour windows.
   - Maps flow cells to sequencer types (NovaSeq X+ / NovaSeq 6000).
   - Maps TOO classes to anatomical groups (e.g. Kidney/Urinary Bladder, Esophagus/Stomach).
3. **Define correctness:**
   - `correct_top1` — true group matches the top-1 prediction.
   - `correct_top2` — true group matches the top-1 or top-2 prediction.
   - The outcome `detected` is set to `correct_top2` (switchable to `correct_top1`).
4. **Fit a logistic regression** of correct TOO prediction against cancer group, stage, and cfDNA yield (with a spline term).
5. **Evaluate** the model using the GLM summary and Wald tests.

## Key settings

- `df["detected"] = df["correct_top2"]` — change to `correct_top1` to evaluate top-1 accuracy instead.
- `yield_col` — set to cfDNA yield per mL of plasma (`tape_col`); can be switched to the Qubit yield (`qubit_col`).
- Optional filter (commented out) to restrict to detection-positive samples only (`score > 0.9906`).
- `group_map` — defines how individual tissue classes are collapsed into anatomical groups.

## Requirements

Python with: `pandas`, `numpy`, `statsmodels`, `patsy`, `matplotlib`.

## Usage

```bash
python too_association_analysis.py
```

Run from a directory where `../InputFiles/` contains the cohort and TOO TSV files.
