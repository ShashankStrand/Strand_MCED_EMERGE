"""
CancerSEEK: Association between Mutant Fragments/mL and Plasma DNA Concentration
Adjusting for and stratifying by Tumor Type and AJCC Stage
"""


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 0. LOAD DATA ──────────────────────────────────────────────────────────────
# Adjust path to your TSV file
df_raw = pd.read_csv("../InputFiles/Bredno.tsv", sep="\t")

# Standardise column names (strip whitespace, replace spaces)
df_raw.columns = [c.strip().replace(" ", "_").replace("/", "_per_") for c in df_raw.columns]
print("Columns detected:", df_raw.columns.tolist())

df_raw.rename(columns={"cfdna_conc_ng_ml": "cfDNA_Yield"}, inplace=True)

# Rename for convenience
detected ="wgbs_classifier_result"
YIELD_COL  = "cfDNA_Yield"
TYPE_COL   = "Source"
STAGE_COL  = "clinical_stage"

# ── 1. CLEAN & FILTER ─────────────────────────────────────────────────────────
df = df_raw.copy()

# Convert numeric columns, coercing errors to NaN
for col in [YIELD_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Keep rows with valid yield AND mutant fragment data, and non-zero mutant frags
# (log transform requires positivity; zero = no mutation detected)
df = df.dropna(subset=[YIELD_COL]).copy()
df = df[df[STAGE_COL].isin(["I","II","III","IV"])]
print(df)

df[detected]=(df[detected]=="detected")

formula = f"""
{detected} ~ 
C({TYPE_COL}) + 
C({STAGE_COL})+
bs({YIELD_COL},df=4)
"""


model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial()
).fit()

#df=df[(df[yield_col] <= df[yield_col].quantile(0.99))]

#X=sm.add_constant(df)
#model = smf.logit(formula, data=X).fit()
#model = smf.logit(formula, data=df).fit_regularized(alpha=1.0)
print(model.summary())
print(model.wald_test_terms())

stages=["I","II","III","IV"]



threshold=0.9906
results = []
for stg in stages:
    dfx = df[df[STAGE_COL] == stg].copy()
    if len(dfx) < 5: continue
    bins = pd.qcut(dfx[YIELD_COL], q=3, duplicates="drop")
    labels = [f"{b.left:.1f}-{b.right:.1f}" for b in bins.cat.categories]
    dfx["yield_tertile"] = pd.Categorical(
        bins.astype(str),
        categories=[str(b) for b in bins.cat.categories],
        ordered=True
    )
    sens = (
        dfx.groupby("yield_tertile")
           .agg(
               n_total=(detected, "size"),
               n_detected=(detected, "sum"),
               median_yield=(YIELD_COL, "median")
           )
    )
    print(sens)
    sens["sensitivity"] = sens["n_detected"] / sens["n_total"]
    sens["se"] = np.sqrt(
        sens["sensitivity"] * (1 - sens["sensitivity"]) / sens["n_total"]
    )
    sens["lcl"] = sens["sensitivity"] - 1.96 * sens["se"]
    sens["ucl"] = sens["sensitivity"] + 1.96 * sens["se"]
    sens = sens.reset_index()
    sens.insert(0, "Stage", stg) 
    sens["yield_tertile"] = pd.Categorical(
        sens["yield_tertile"],
        #categories=["Low", "Mid", "High"],
        ordered=True
    )
    results.append(sens)

final_df = pd.concat(results, ignore_index=True)
stage_order = ["I", "II", "III","IV"]

final_df["Stage"] = pd.Categorical(
    final_df["Stage"],
    categories=stage_order,
    ordered=True
)

final_df = final_df.sort_values(["Stage", "yield_tertile"])

# -----------------------------
# Output
# -----------------------------
print("\nFinal combined dataframe:")
print(final_df)
final_df.to_csv("../Tables/Bredno_Stage_wise_Tertile_Sens.tsv",sep="\t")
