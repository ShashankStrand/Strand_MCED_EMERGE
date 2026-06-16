"""
CancerSEEK: Association between Mutant Fragments/mL and Pasma DNA Concentration
Adjusting for and stratifying by Tumor Type and AJCC Stagge
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
df_raw = pd.read_csv("../InputFiles/CancerSeek-S5.tsv", sep="\t")

# Standardise column names (strip whitespace, replace spaces)
df_raw.columns = [c.strip().replace(" ", "_").replace("/", "_per_") for c in df_raw.columns]
print("Columns detected:", df_raw.columns.tolist())

df_raw.rename(columns={"Plasma_DNA_concentration_(ng_per_mL)": "cfDNA_Yield"}, inplace=True)

# Rename for convenience
detected ="CancerSEEK_Test_Result"
YIELD_COL  = "cfDNA_Yield"
MUT_COL    = "Mutant_fragments_per_mL_plasma"
MUT_COL    = "Mutant_allele_frequency_(%)"
MAF_COL    = "Mutant_allele_frequency_(%)"
TYPE_COL   = "Tumor_type"
STAGE_COL  = "AJCC_Stage"
TUM_COL  = "In_Tumor"


print(f"\nUsing columns:\n  Yield: {YIELD_COL}\n  Mutant frags: {MUT_COL}\n  MAF: {MAF_COL}\n  Type: {TYPE_COL}\n  Stage: {STAGE_COL}\n Tumor: {TUM_COL}")

# ── 1. CLEAN & FILTER ─────────────────────────────────────────────────────────
df = df_raw.copy()

#df=df[df[TUM_COL]=="Present"]

# Convert numeric columns, coercing errors to NaN
for col in [YIELD_COL, MUT_COL, MAF_COL]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


print(df)
   
# Keep rows with valid yield AND mutant fragment data, and non-zero mutant frags
# (log transform requires positivity; zero = no mutation detected)
#df = df.dropna(subset=[YIELD_COL, MUT_COL]).copy()
df = df[df[STAGE_COL].isin(["I","II","III","IV"])]
print(df)

# Normalise stage labels
STAGE_ORDER = ["I","II","III"]
print(df)

df[detected]=(df[detected]=="Positive")



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

stages=["I","II","III"]

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
stage_order = ["I", "II", "III"]

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
final_df.to_csv("../Tables/CancerSeek_Stage_wise_Tertile_Sens.tsv",sep="\t")
