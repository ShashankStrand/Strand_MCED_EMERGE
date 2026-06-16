import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import bs
import matplotlib.pyplot as plt


import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings("ignore")

# Read the TSV file
#import the two Supplementary Table 1 tabs
df = pd.read_csv("../InputFiles/cohort-main.tsv", sep="\t")
bdf = pd.read_csv("../InputFiles/Benign.tsv", sep="\t")
df = df.rename(columns={'Plasma Storage Time': 'plasma_storage_time'})
bdf = bdf.rename(columns={'Plasma Storage Time': 'plasma_storage_time'})

#import Supplementary Table 5 tabs
toocvdf = pd.read_csv("../InputFiles/TOO-Ge80Lt20FragTop300-CV.tsv", sep="\t")
tootestdf = pd.read_csv("../InputFiles/TOO-Ge80Lt20FragTop300-Test.tsv", sep="\t")
toobenigndf = pd.read_csv("../InputFiles/TOO-Ge80Lt20FragTop300-Benign.tsv", sep="\t")
tootestdf=tootestdf[["Sample_id","True_Class","Top1_Class","Top1_Prob","Top2_Class","Top2_Prob","Top3_Class","Top3_Prob"]]
toocvdf=toocvdf[["Sample_id","True_Class","Top1_Class","Top1_Prob","Top2_Class","Top2_Prob","Top3_Class","Top3_Prob"]]
toobenigndf=toobenigndf[["Sample_id","True_Class","Top1_Class","Top1_Prob","Top2_Class","Top2_Prob","Top3_Class","Top3_Prob"]]
toodf = pd.concat([toocvdf, tootestdf,toobenigndf], ignore_index=True)

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
qubit_col = "yield_ml_plasma_ng"   # adjust if different
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
split_col="Random_Split"
# -------------------------------------------

yield_col=tape_col
#yield_col=qubit_col

cols = [site_col,pst_col, yield_col, cancer_col, stage_col, sample_col, score_col, bin_col, age_col, sex_col, tob_col,alc_col,fc_col, bmi_col, dna_inp_col,cov_col, cal_score_col, frag_cnt,unalgn_pcnt_col,coltime_col,split_col]
df = df.reindex(columns=cols)
bdf = bdf.reindex(columns=cols)
combined = pd.concat([df, bdf], ignore_index=True)
combined[yield_col] = pd.to_numeric(combined[yield_col], errors="coerce")
combined[bmi_col] = pd.to_numeric(combined[bmi_col], errors="coerce")

#combine too results with cohort details
toodf["Sample_id_clean"] = toodf["Sample_id"].str.split("_", n=1).str[1]
combined = combined.merge(toodf, left_on="sample", right_on="Sample_id_clean", how="left")
df=combined


df = df.rename(columns={
    "DNA_Input_For_Lib_Prep (ng)": "DNA_Input_For_Lib_Prep",
    yield_col: "cfDNAyield",
    "cov.avg":"coverage",
    coltime_col:"Collection_Time"
})

yield_col = "cfDNAyield"   # yield is reserved
stage_col = "Stage"
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


df = df.copy()
#99% specificity threshold in training
threshold = 0.9906
stglist=["Benign","I","II","III","IV"]


df=df[df[stage_col]!="control"]
#df=df[df[stage_col]!="Benign"]

#map too output to anatomical groups
group_map = {
    "Prostate": ["Prostate"],
    "Kidney,Urinary Bladder": ["Kidney", "Urinary Bladder"],
    "Lung": ["Lung"],
    "Esophagus,Stomach": ["Esophagus", "Stomach"],
    "Colorectal": ["Colorectal"],
    "Cervix": ["Cervix"],
    "Breast": ["Breast"],
    "Uterus,Ovary": ["Uterus", "Ovary"],
    "Liver,Pancreas,Gall Bladder": ["Liver", "Pancreas", "Gall Bladder"]
}


# invert mapping
class_to_group = {
    cls: group
    for group, lst in group_map.items()
    for cls in lst
}

# -----------------------------
# Apply grouping
# -----------------------------
df["True_Group"] = df["True_Class"].map(class_to_group)
df["Top1_Group"] = df["Top1_Class"].map(class_to_group)
df["Top2_Group"] = df["Top2_Class"].map(class_to_group)

# drop missing mappings (important)
df = df.dropna(subset=["True_Group", "Top1_Group", "Top2_Group"])

# -----------------------------
# Correctness definitions
# -----------------------------
df["correct_top1"] = (df["Top1_Group"] == df["True_Group"]).astype(int)
df["correct_top2"] = (
        (df["Top1_Group"] == df["True_Group"]) |
        (df["Top2_Group"] == df["True_Group"])
    ).astype(int)

#EDIT this to too1 or too2 as needed
df["detected"] = df["correct_top2"]

#keep only benign and independent validation
df=df[df[split_col]!="LeaveIn"]

#turn this on if we need to only consider samples which are detected positive
#df=df[df[score_col]>0.9906]

cancer_col="True_Group"

# -----------------------------
# 3. CLEAN DATA
# -----------------------------
df = df.dropna(subset=[yield_col, bmi_col, stage_col, cancer_col, "detected"])
print(len(df))

formula = f"""
detected ~ 
C({cancer_col}) + 
C({stage_col})+
bs({yield_col},df=4)
"""

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial()
).fit()

print(model.summary())
print(model.wald_test_terms())


