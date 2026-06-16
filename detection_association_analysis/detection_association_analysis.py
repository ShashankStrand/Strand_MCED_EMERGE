from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import bs
import matplotlib.pyplot as plt


import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings("ignore")


def auc_ci_bootstrap(y, y_pred, n_bootstraps=1000, alpha=0.95):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y), len(y))
        
        if len(np.unique(y[indices])) < 2:
            continue
        
        score = roc_auc_score(y[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    
    lower = sorted_scores[int((1 - alpha)/2 * len(sorted_scores))]
    upper = sorted_scores[int((1 + alpha)/2 * len(sorted_scores))]
    med = sorted_scores[int(0.5 * len(sorted_scores))]
    return lower, upper, med


def forest(model):
    # Extract coefficients
    coef = model.params
    conf = model.conf_int()
    conf.columns = ["lower", "upper"]

    # Convert to odds ratios
    or_df = pd.DataFrame({
        "OR": np.exp(coef),
        "LowerCI": np.exp(conf["lower"]),
        "UpperCI": np.exp(conf["upper"])
    })

    # Remove intercept if desired
    or_df = or_df.drop("Intercept")

    # Forest plot
    fig, ax = plt.subplots(figsize=(6,4))

    ax.errorbar(
        or_df["OR"],
        or_df.index,
        xerr=[
            or_df["OR"] - or_df["LowerCI"],
            or_df["UpperCI"] - or_df["OR"]
        ],
        fmt='o'
    )

    ax.axvline(1, color='gray', linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_ylabel("Variable")

    plt.tight_layout()
    plt.show()



# Read the TSV file
# This is part of Supplementary Table 1, the main cohort tab
df = pd.read_csv("../InputFiles/cohort-main.tsv", sep="\t")
# This is part of Supplementary Table 1, the Benign cohort tab 
bdf = pd.read_csv("../InputFiles/Benign.tsv", sep="\t")

#Restrict to Independent Validation Set + Benign
df=df[df["Random_Split"]=="LeaveOut"]


df = df.rename(columns={'Plasma Storage Time': 'plasma_storage_time'})
bdf = bdf.rename(columns={'Plasma Storage Time': 'plasma_storage_time'})

# ---- CHANGE THESE COLUMN NAMES IF NEEDED ----
coltime_col="Collection Time"
tape_col="cfDNA_yield_per_ml_of_plasma_pg_ml"
unalgn_pcnt_col="Unaligned Read %"
frag_cnt="preres_nOriginalPatrRows"
cal_score_col="Calibrated_Score"
cov_col="cov.avg"
dna_inp_col="DNA_Input_For_Lib_Prep (ng)"
age_col="Age"
pst_col='plasma_storage_time'
site_col="Site"
qubit_col = "yield_ml_plasma_ng"
cancer_col = "Cancer_type"
stage_col = "Stage"
sample_col = "sample"
score_col="RandomSplit_Score"
bin_col="PST_bin"
sex_col="Gender"
tob_col="Tobacco"
alc_col="Alcohol"
fc_col="Flow_cell"
bmi_col="BMI"
# -------------------------------------------

yield_col=tape_col
#yield_col=qubit_col

cols = [site_col,pst_col, yield_col, cancer_col, stage_col, sample_col, score_col, bin_col, age_col, sex_col, tob_col,alc_col,fc_col, bmi_col, dna_inp_col,cov_col, cal_score_col, frag_cnt,unalgn_pcnt_col,coltime_col]

df = df.reindex(columns=cols)
bdf = bdf.reindex(columns=cols)

#combine main+benign cohorts
combined = pd.concat([df, bdf], ignore_index=True)
combined[yield_col] = pd.to_numeric(combined[yield_col], errors="coerce")
combined[bmi_col] = pd.to_numeric(combined[bmi_col], errors="coerce")

#check counts
#print(combined.groupby(stage_col).size())

df=combined
df = df.rename(columns={
    "DNA_Input_For_Lib_Prep (ng)": "DNA_Input_For_Lib_Prep",
    yield_col: "cfDNAyield",
    "cov.avg":"coverage",
    coltime_col:"Collection_Time"
})
yield_col = "cfDNAyield"   # yield is reserved
dna_inp_col="DNA_Input_For_Lib_Prep"
cov_col="coverage"
coltime_col="Collection_Time"


#bin collection time
# Convert to datetime
df[coltime_col] = pd.to_datetime(df[coltime_col], format='%I:%M:%S %p')
# Extract hour
hour = df[coltime_col].dt.hour
# Create 3-hour bins
labels = ['6-9', '9-12', '12-15', '15-18', '18-21', '21-24']
bins = [6, 9, 12, 15, 18, 21, 24]
df[coltime_col] = pd.cut(
    hour,
    bins=bins,
    labels=labels,
    right=False
)

#map flow cells to sequencers
mapping = {
    '25B': 'NovaSeq X+',
    '10B': 'NovaSeq X+',
    'S4': 'NovaSeq 6000'
}
df[fc_col] = df[fc_col].map(mapping)

#print(df)

#for rare tissue types (only relevant for benign, replace by other)
counts = df[cancer_col].value_counts()
rare = counts[counts < 3].index
df[cancer_col] = df[cancer_col].replace(rare, "Other")
print(df.groupby(cancer_col).size())


# -------------------------------------------

df = df.copy()
#print(df)

#99% specificity threshold in training
threshold = 0.9906
stglist=["Benign","I","II","III","IV"]


#set up the outcome variable
df=df[df[stage_col]!="control"]
#df=df[df[stage_col]!="Benign"]
#df=df[df[dna_inp_col]==20]
#print(df[stage_col].unique())
df["detected"] = (df[score_col] > threshold).astype(int)

n=len(df)

#drop missing values
df = df.dropna(subset=[yield_col,  bmi_col,"detected"])
m=len(df)
print("Rows lost =",n-m)

# -----------------------------
# 4. FIT MODEL
# (with nonlinearity + interaction)
# -----------------------------

'''
formula = f"""
detected ~ 
C({cancer_col}) + 
C({stage_col}) + 
bs({yield_col}, df=4 )+ 
C({stage_col}):{yield_col}
"""
'''

formula = f"""
detected ~ 
C({cancer_col}) + 
C({stage_col})+
bs({yield_col},df=4)+
{age_col}+
C({sex_col})+
C({tob_col})+
C({alc_col})+
C({fc_col})+
{pst_col}+
{bmi_col}+
{dna_inp_col}+
{cov_col}
"""

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial()
).fit()

print(model.summary())
print(model.wald_test_terms())

#forest(model)

print(sm.__version__)

#AUC
y_pred_proba = model.predict(df)
print(np.isnan(y_pred_proba).sum())
mask = ~np.isnan(y_pred_proba) & ~df["detected"].isna()
auc_low, auc_high, auc_med = auc_ci_bootstrap(df["detected"].values[mask], y_pred_proba.values[mask]) 
print("AUC",auc_low,auc_high, auc_med)

