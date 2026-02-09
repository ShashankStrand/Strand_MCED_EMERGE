from statsmodels.stats.proportion import proportion_confint
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ========== PARAMETERS — adapt to your data ==========
LABEL_COL = "type"     # column indicating sample label (e.g. "control" vs cancer types)  
CONTROL_LABEL = "control"  # how control samples are labelled
SCORE_COL = "GenCncrs"    # model output (probability or score) for positive class (cancer)
STAGE_COL = "stage"    # column indicating cancer stage (if exists)
TYPE_COL = "type"  # column indicating cancer type (if exists)

N_BOOTSTRAPS = 1000     # number of bootstrap replicates for CI
CI_PCT = 95             # CI confidence level (e.g. 95 → 2.5th–97.5th percentiles)

#=========================  FUNCTIONS ==================


def compute_confusion_metrics(df, threshold=0.5):
    df = df.copy()
    df["y_pred"] = (df["y_score"] >= threshold).astype(int)
    TP = ((df["y_true"] == 1) & (df["y_pred"] == 1)).sum()
    FN = ((df["y_true"] == 1) & (df["y_pred"] == 0)).sum()
    TN = ((df["y_true"] == 0) & (df["y_pred"] == 0)).sum()
    FP = ((df["y_true"] == 0) & (df["y_pred"] == 1)).sum()
    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    return sens, spec, TP, FN, TN, FP

'''
def bootstrap_subgroup_sens_spec(df_subgroup, threshold=0.5, n_boot=1000):
    boot_sens = []
    boot_spec = []
    for i in range(n_boot):
        bs = df_subgroup.sample(n=len(df_subgroup), replace=True)
        TP = sum((bs.y_true == 1) & (bs.y_score >= threshold))
        FN = sum((bs.y_true == 1) & (bs.y_score < threshold))
        TN = sum((bs.y_true == 0) & (bs.y_score < threshold))
        FP = sum((bs.y_true == 0) & (bs.y_score >= threshold))
        if (TP + FN) == 0 or (TN + FP) == 0:
            continue  # skip invalid bootstrap
        boot_sens.append(TP / (TP + FN))
        boot_spec.append(TN / (TN + FP))
    sens_ci = np.percentile(boot_sens, [2.5, 97.5])
    spec_ci = np.percentile(boot_spec, [2.5, 97.5])
    return {"sens_ci": sens_ci, "spec_ci": spec_ci}
'''

def subgroup_sens_spec_ci(df_subgroup, threshold=0.5, alpha=0.05, method="wilson"):
    """
    Compute point estimates + binomial-proportion CI (sensitivity & specificity)
    for a subgroup (df_subgroup must include both positives and negatives).
    
    Args:
      df_subgroup: pandas DataFrame with columns y_true (0/1), y_score (continuous)
      threshold: float, cutoff to classify as positive
      alpha: float, significance level for CI (default 0.05 → 95% CI)
      method: str, method for CI: “wilson”, “beta” (Clopper–Pearson), “jeffreys”, “agresti_coull”, etc.
    
    Returns:
      dict with:
        sens: float (sensitivity point estimate)
        sens_ci: (lower, upper) CI bounds
        spec: float (specificity point estimate)
        spec_ci: (lower, upper) CI bounds
        TP, FN, TN, FP: counts
    """
    # Binarize predictions
    df = df_subgroup.copy()
    df["y_pred"] = (df["y_score"] >= threshold).astype(int)
    
    TP = int(((df["y_true"] == 1) & (df["y_pred"] == 1)).sum())
    FN = int(((df["y_true"] == 1) & (df["y_pred"] == 0)).sum())
    TN = int(((df["y_true"] == 0) & (df["y_pred"] == 0)).sum())
    FP = int(((df["y_true"] == 0) & (df["y_pred"] == 1)).sum())
    
    # Compute point estimates
    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    
    # Confidence intervals using binomial proportion CI
    sens_ci = (np.nan, np.nan)
    spec_ci = (np.nan, np.nan)
    
    if (TP + FN) > 0:
        sens_ci = proportion_confint(
            count=TP, nobs=(TP + FN), alpha=alpha, method=method
        )
    if (TN + FP) > 0:
        spec_ci = proportion_confint(
            count=TN, nobs=(TN + FP), alpha=alpha, method=method
        )

    return {
        "TP": TP,
        "FN": FN,
        "TN": TN,
        "FP": FP,
        "sensitivity": sens,
        "sens_ci": [sens_ci[0],sens_ci[1]],
        "specificity": spec,
        "spec_ci": [spec_ci[0],spec_ci[1]]
    }

def bootstrap_subgroup_sens_spec_ci(df_subgroup, onlySens, threshold, n_boot=1000, min_valid=10, random_state=None):
    """
    Bootstrap sensitivity & specificity for a subgroup DataFrame.
    Args:
      df_subgroup: pandas DataFrame containing both positive (cancer) and negative (control) rows.
      threshold: float, cutoff on score to call positive
      n_boot: int, number of bootstrap replicates
      min_valid: int, minimum number of valid bootstrap replicates needed to compute CI
      random_state: int or None
    Returns:
      dict with keys:
        sens_ci: (lower, upper) if valid; else (nan, nan)
        spec_ci: same
        n_boot_valid: number of bootstraps used
        note: status / warning message
    """
    rng = np.random.RandomState(random_state)
    boot_sens = []
    boot_spec = []
    n = len(df_subgroup)
    #print("n=",n,df_subgroup,threshold)
    for i in range(n_boot):
        bs = df_subgroup.sample(n=n, replace=True, random_state=rng.randint(0, 1_000_000))
        #print(df_subgroup['type'].unique(),len(bs))
        # compute confusion
        y_true = bs["y_true"].values
        y_score = bs["y_score"].values
        y_pred = (y_score >= threshold).astype(int)

        # check both classes present
        if not(onlySens):
            if len(np.unique(y_true)) < 2:
                continue

        # compute sens/spec
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        #print(TP,FN,TN,FP)
        
        # skip if denominators zero
        if onlySens:
            if (TP + FN) == 0:
                continue
        else:
            if ((TP + FN) == 0) or ((TN + FP) == 0):
                continue

        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        boot_sens.append(sens)
        boot_spec.append(spec)

    n_valid = len(boot_sens)
    #print(n_valid)
    result = {"n_boot_valid": n_valid}

    if n_valid < min_valid:
        result["sens_ci"] = (np.nan, np.nan)
        result["spec_ci"] = (np.nan, np.nan)
        result["note"] = f"Too few valid bootstraps ({n_valid} < {min_valid}); CI unreliable"
    else:
        sens_ci = np.percentile(boot_sens, [2.5, 97.5])
        spec_ci = np.percentile(boot_spec, [2.5, 97.5])
        result["sens_ci"] = (sens_ci[0], sens_ci[1])
        result["spec_ci"] = (spec_ci[0], spec_ci[1])
        result["note"] = "OK"
    return result


def roc_auc_ci(df,thr):
    # ========== 2) Compute ROC + AUC on the full data ==========
    y_true = df["y_true"].values
    y_score = df["y_score"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc0 = roc_auc_score(y_true, y_score)
    #print("AUC:", auc0)

    # ========== 3) Bootstrap to get CIs for AUC (and optionally Sens/Spec) ==========
    boot_aucs = []
    boot_sens = []
    boot_spec = []
    rng = np.random.RandomState(123)

    for i in range(N_BOOTSTRAPS):
        idx = rng.choice(len(df), size=len(df), replace=True)
        df_bs = df.iloc[idx]
        # need at least one positive & one negative case in bootstrap sample
        if len(np.unique(df_bs["y_true"])) < 2:
            continue
        # AUC
        boot_aucs.append(roc_auc_score(df_bs["y_true"], df_bs["y_score"]))
        # (Optional) Sens/Spec at threshold 0.5
        bs_sens, bs_spec, _, _, _, _ = compute_confusion_metrics(df_bs, thr)
        boot_sens.append(bs_sens)
        boot_spec.append(bs_spec)

    auc_ci = np.percentile(boot_aucs, [(100 - CI_PCT) / 2, 100 - (100 - CI_PCT) / 2])
    sens_ci = np.percentile(boot_sens, [(100 - CI_PCT) / 2, 100 - (100 - CI_PCT) / 2])
    spec_ci = np.percentile(boot_spec, [(100 - CI_PCT) / 2, 100 - (100 - CI_PCT) / 2])

    #print(auc_ci)
    #print(f"AUC {CI_PCT}% CI:", auc_ci)
    #print(f"Sensitivity {CI_PCT}% CI:", sens_ci)
    #print(f"Specificity {CI_PCT}% CI:", spec_ci)


    tprs = []
    for i in range(N_BOOTSTRAPS):
            idx = rng.choice(len(df), size=len(df), replace=True)
            df_bs = df.iloc[idx]
            if len(np.unique(df_bs["y_true"])) < 2:
                continue
            fpr_bs, tpr_bs, _ = roc_curve(df_bs["y_true"], df_bs["y_score"])
            # interpolate tpr_bs at base fpr points
            tpr_interp = np.interp(fpr, fpr_bs, tpr_bs)
            tprs.append(tpr_interp)

    tprs = np.array(tprs)
    lower_tpr = np.percentile(tprs, (100 - CI_PCT) / 2, axis=0)
    upper_tpr = np.percentile(tprs, 100 - (100 - CI_PCT) / 2, axis=0)
    #print( lower_tpr, upper_tpr)
    return(auc0,auc_ci,tpr,lower_tpr,upper_tpr,fpr)


def addtoresult(r,e,t,tv,s,sens,spec,sens_ci_l,sens_ci_u,spec_ci_l,spec_ci_u,tp,fn,tn,fp,auc,auc_ci_l,auc_ci_u):
    r.append({
        "split": fullname(e),
        "type": t,
        "mode": fullname(tv),
        "stage": s,
        "sensitivity": sens,
        "specificity": spec,
        "sensitivity_ci_l": sens_ci_l,
        "sensitivity_ci_u": sens_ci_u,        
        "specificity_ci_l": spec_ci_l,
        "specificity_ci_u": spec_ci_u,
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
        "auc": auc, "auc_ci_l": auc_ci_l, "auc_ci_u": auc_ci_u
    })
    return(r)

def saveImage(f,t,ci_l,ci_u,auc,auc_ci,e,mode):
    plt.figure(figsize=(6,6))
    plt.plot(f, t, label=f'ROC (AUC = {auc:.3f}, 95% CI [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}])', color='blue')
    plt.fill_between(f, ci_l, ci_u, color='blue', alpha=0.2,
                 label=f'{CI_PCT}% CI band')
    plt.plot([0,1], [0,1], '--', color='grey')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(fullname(e)+' Split: '+mode+'\n  ROC curve with bootstrap CI band')
    plt.legend(loc='lower right')
    plt.grid(True)
    #plt.show()
    plt.savefig("../SuppFigs/"+fullname(e)+'_Split_'+mode+'_ROC_curve.png')
    

def calculate(cutoffs,expts,mode,path,roc):
    results = []
    for e in expts:
        df = pd.read_csv(path+e+mode+".tsv",sep="\t")
        df["y_true"] = np.where(df[LABEL_COL] == CONTROL_LABEL, 0, 1)
        df["y_score"] = df[SCORE_COL]
        #df['stage'] = df['stage'].str.replace(r'\bII\b', 'I', regex=True)
        thr=cutoffs[e]

        #SUBSEC A: OVERALL
        #use only non benign samples from the main cohort
        #other cohorts do not have benign, those are labeled non-malignant growth
        #compute sens and spec and TP, TN, FP, FN
        r1 = compute_confusion_metrics(df[df['stage']!='Benign'], thr)

        #for the non-main cohorts only sensitivity because they have no controls
        if ((mode=="Rare") or (mode=="Benign")): onlySens=True
        else: onlySens=False

        #Compute CI for sens and spec using bootstrap, again non benign samples only
        r2 =bootstrap_subgroup_sens_spec_ci(df[df['stage']!='Benign'],onlySens,thr)

        #compute AUC and 95% CI, again non-benign only 
        if roc:
            auc0,auc_ci,tpr,lower_tpr,upper_tpr,fpr=roc_auc_ci(df[df['stage']!='Benign'],thr)
            saveImage(fpr,tpr,lower_tpr,upper_tpr,auc0,auc_ci,e,mode)
        else: auc0,auc_ci,tpr,lower_tpr,upper_tpr,fpr=0,[0,0],0,0,0,0
        results=addtoresult(results,e, "All", mode , "All", r1[0], r1[1], r2['sens_ci'][0], r2['sens_ci'][1],r2['spec_ci'][0], r2['spec_ci'][1], r1[2], r1[3], r1[4], r1[5],auc0, auc_ci[0],auc_ci[1])

        #SUBSEC B: STAGEWISE
        df_controls = df[df[LABEL_COL] == CONTROL_LABEL]
        for stage, df_cancer_sub in df[df[LABEL_COL] != CONTROL_LABEL].groupby([STAGE_COL]):
            df_sub = pd.concat([df_cancer_sub, df_controls], ignore_index=True)
            r1 = compute_confusion_metrics(df_sub, thr)
            r2 =bootstrap_subgroup_sens_spec_ci(df_sub,onlySens,thr)
            results=addtoresult(results,e, "All", mode, stage[0], r1[0], r1[1], r2['sens_ci'][0], r2['sens_ci'][1],r2['spec_ci'][0], r2['spec_ci'][1], r1[2], r1[3], r1[4], r1[5],"", "","")

        #SUBSEC C: CANCERWISE, STAGEWISE
        #CIs computed using wilson rather than bootstrap here
        df_controls = df[df[LABEL_COL] == CONTROL_LABEL]
        #df_e=df[df['stage'].isin(["I","II"])]
        #for ctype, df_cancer_sub in df_e[df_e[LABEL_COL] != CONTROL_LABEL].groupby([TYPE_COL]):
        for ctype, df_cancer_sub in df[df[LABEL_COL] != CONTROL_LABEL].groupby([TYPE_COL]):
            df_sub = pd.concat([df_cancer_sub, df_controls], ignore_index=True)
            r1 = compute_confusion_metrics(df_sub[df_sub['stage']!='Benign'], thr)
            #r2 =bootstrap_subgroup_sens_spec_ci(df_sub[df_sub['stage']!='Benign'],thr)
            r2 =subgroup_sens_spec_ci(df_sub[df_sub['stage']!='Benign'],thr)
            results=addtoresult(results,e, ctype[0], mode, "All", r1[0], r1[1], r2['sens_ci'][0], r2['sens_ci'][1],r2['spec_ci'][0], r2['spec_ci'][1], r1[2], r1[3], r1[4], r1[5],"", "","")
        for (ctype,stage), df_cancer_sub in df[df[LABEL_COL] != CONTROL_LABEL].groupby([TYPE_COL,STAGE_COL]):
            df_sub = pd.concat([df_cancer_sub, df_controls], ignore_index=True)
            r1 = compute_confusion_metrics(df_sub, thr)
            r2=subgroup_sens_spec_ci(df_sub,thr)
            results=addtoresult(results,e, ctype, mode, stage, r1[0], r1[1], r2['sens_ci'][0], r2['sens_ci'][1],r2['spec_ci'][0], r2['spec_ci'][1], r1[2], r1[3], r1[4], r1[5],"", "","")
    return(results)

def fullname(e):
    fn={'Random':'Random','Site':'By Collection Site (Controls randomized)','Coll':'By Collection Date (Controls randomized)','Seq':'By Sequencing Batch (Controls randomized)','Seq10':'By Sequencing Batch (10 Cancers Only, Controls randomized)',"CV":'Training (Cross-Validation)',"Test":"Independent Validation","Rare":"Rare Cancers","Benign":"Non-malignant Growth",'PreRescueTest':'Without Rescue','AltSite':'By Collection Site','AltColl':'By Collection Date','AltSeq':'By Sequencing Batch','AltSeq10':'By Sequencing Batch (10 Cancers Only)',"HeadAndNeck":"Head","Single-":"Single Cancer Model", "BreastTest":"Breast", "LungTest":"Lung", "CervixTest":"Cervix", "ProstateTest":"Prostate", "ColorectalTest":"Colorectal", "HeadAndNeckTest":"Head And Neck"}
    return(fn[e])


# ========== MAIN ==========


top10=['Breast','Lung','control','Pancreas','Cervix','Colorectal','Stomach','Lung','Ovary','Gall Bladder','Liver']
expts=['Single-','Random','Site','Coll','Seq','Seq10','AltSite','AltColl','AltSeq','AltSeq10']
cutoffs={}
spec=[]
score='GenCncrs'
path="../InputFiles/"

#calculate specificity cutoffs for 99% training specificity
#or for 90%, toggle appropriately
#scutoff=.99
#scutoff_name="0.99"
scutoff=.90
scutoff_name="0.90"

for e in expts[1:]:
   df = pd.read_csv(path+e+"CV-80.tsv",sep="\t")
   cutoffs[e]=df[df[TYPE_COL]=='control']['score'].quantile(scutoff)
print(cutoffs)

'''
#hardcode cutoffs for single tests calculated elsewhere, not needed anymore
cutoffBreast={'Single-': 0.1226}
cutoffCervix={'Single-': 0.0689}
cutoffProstate={'Single-': 0.0785}
cutoffLung={'Single-': 0.0566}
cutoffColorectal={'Single-': 0.2079}
cutoffHeadAndNeck={'Single-': 0.4261}
'''

results1=calculate(cutoffs,expts[1:],"CV",path,True)
results2=calculate(cutoffs,expts[1:],"Test",path,True)
results3=calculate(cutoffs,[expts[1]],"Rare",path,False)
results4=calculate(cutoffs,[expts[1]],"PreRescueTest",path,True)
results5=calculate(cutoffs,[expts[1]],"Benign",path,False)

'''
results7=calculate(cutoffBreast,[expts[0]],"BreastTest",path,False)
results8=calculate(cutoffLung,[expts[0]],"LungTest",path,False)
results9=calculate(cutoffColorectal,[expts[0]],"ColorectalTest",path,False)
results10=calculate(cutoffProstate,[expts[0]],"ProstateTest",path,False)
results11=calculate(cutoffCervix,[expts[0]],"CervixTest",path,False)
results12=calculate(cutoffHeadAndNeck,[expts[0]],"HeadAndNeckTest",path,False)
'''

df_res1 = pd.DataFrame(results1)
df_res2 = pd.DataFrame(results2)
df_res3 = pd.DataFrame(results3)
df_res4 = pd.DataFrame(results4)
df_res5 = pd.DataFrame(results5)

'''
df_res7 = pd.DataFrame(results7)
df_res8 = pd.DataFrame(results8)
df_res9 = pd.DataFrame(results9)
df_res10 = pd.DataFrame(results10)
df_res11 = pd.DataFrame(results11)
df_res12 = pd.DataFrame(results12)
'''

df_res = pd.concat([df_res1, df_res2], axis=0)
df_res = pd.concat([df_res, df_res3], axis=0)
df_res = pd.concat([df_res, df_res4], axis=0)
df_res = pd.concat([df_res, df_res5], axis=0)


'''
df_res = pd.concat([df_res, df_res7], axis=0)
df_res = pd.concat([df_res, df_res8], axis=0)
df_res = pd.concat([df_res, df_res9], axis=0)
df_res = pd.concat([df_res, df_res10], axis=0)
df_res = pd.concat([df_res, df_res11], axis=0)
df_res = pd.concat([df_res, df_res12], axis=0)
'''

df_res.to_csv("../Tables/Confidence Intervals-"+scutoff_name+".tsv",sep="\t")
with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.max_colwidth', None,
    'display.width', None
): print(df_res)




'''

# ========== 4) Plot ROC + bootstrap CI-band + annotate AUC + CI ==========
# First collect TPR arrays from bootstraps

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc0:.3f})', color='blue')
plt.fill_between(fpr, lower_tpr, upper_tpr, color='blue', alpha=0.2,
                 label=f'{CI_PCT}% CI band')
plt.plot([0,1], [0,1], '--', color='grey')
#plt.text(0.6, 0.05,
#         f'AUC = {auc0:.3f}\n{CI_PCT}% CI = [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]',
#         bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC curve with bootstrap CI band')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig("my_plot.png")
'''
