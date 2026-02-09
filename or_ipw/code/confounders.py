import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def compute_smd(df, cov, binary_outcome, weight_col=None):
    treated = df[binary_outcome]==1
    control = ~treated

    if weight_col is None:
        m_t = df.loc[treated, cov].mean()
        m_c = df.loc[control, cov].mean()
        sd_t = df.loc[treated, cov].std()
        sd_c = df.loc[control, cov].std()
    else:
        w_t = df.loc[treated, weight_col]
        w_c = df.loc[control, weight_col]
        m_t = np.average(df.loc[treated, cov], weights=w_t)
        m_c = np.average(df.loc[control, cov], weights=w_c)
        sd_t = np.sqrt(np.average((df.loc[treated,cov]-m_t)**2, weights=w_t))
        sd_c = np.sqrt(np.average((df.loc[control,cov]-m_c)**2, weights=w_c))

    pooled = np.sqrt((sd_t**2 + sd_c**2)/2)
    return (m_t - m_c)/pooled


def LR(d,l,o):
   X=d[l]
   y=d[o]
   X = sm.add_constant(X)  # adds intercept
   X = X.astype(float) 
   #model = sm.Logit(y, X).fit()
   model = sm.GLM(y, X,family=sm.families.Binomial(),freq_weights=None).fit()
   y_pred_proba = model.predict(X)
   auc = roc_auc_score(y, y_pred_proba)
   params = model.params              # log-odds (beta) for each predictor (and intercept)
   pvals  = model.pvalues             # p-values
   conf   = model.conf_int()          # 95% CI on log-odds (lower, upper bounds)
   print(params)
   or_vals = np.exp(params)
   or_lower = np.exp(conf[0])
   or_upper = np.exp(conf[1])
   df_or = pd.DataFrame({
       "Covariate" : params.index.values,
       "OR"       : or_vals,
       "OR_lower" : or_lower,
       "OR_upper" : or_upper,
       "p_value"  : pvals,
       "AUC" : [auc]*len(or_vals),
       "Covariates" : [l]*len(or_vals),
       
   })
   #df_or = df_or.round(3)
   # Example: save to CSV for inclusion in manuscript/table
   #print("==================")
   #print(df_or)
   return(df_or)

def clean(x):
    a=0
    b=0
    if 0 in x.index: a=x.loc[0]
    if 1 in x.index: b=x.loc[1]    
    return([a,b])


def computePVal(df,k,c,o,n):
    results = []

    for group in df[o].unique():
        subset = df[df[o] == group]
        k_0 = subset[subset[c] == 0][k]
        k_1 = subset[subset[c] == 1][k]
        if (len(k_0)>1) & (len(k_1)>1):
            mean_0 = k_0.mean()
            std_0 = k_0.std()
            mean_1 = k_1.mean()
            std_1 = k_1.std()
            stat, pval = mannwhitneyu(k_0, k_1, alternative='two-sided')
            results.append({
                "name": n,
                "score": k,
                "confounder": c,
                'o': group,
                'mean_c0': mean_0,
                'std_c0': std_0,
                'mean_c1': mean_1,
                'std_c1': std_1,
                'p_value': pval
        })
    return(results)


#Read the file with all covariates
d1=pd.read_csv("../InputFiles/Cohort1.tsv",parse_dates=["Collection Date","Sequencing Date","Extraction Date"],sep="\t")
d1['Collection Date'] = pd.to_datetime(d1['Collection Date'], errors='coerce', dayfirst=True)
d1['Sequencing Date'] = pd.to_datetime(d1['Sequencing Date'], errors='coerce', dayfirst=True)
d1['Extraction Date'] = pd.to_datetime(d1['Extraction Date'], errors='coerce', dayfirst=True)

#Binarize 
d1['Gender']=np.where(d1['Gender'] != 'Male', 0, 1)
d1['Tobacco']=np.where(d1['Tobacco'] != 'No Habit', 0, 1)
d1['Outcome']=np.where(d1['Cancer_type'] != 'control', 1, 0)
d1['FC']=np.where(d1['Flow_cell'].str.contains("B"), 1, 0)
d1['Bin_Age']=np.where(d1['Age']>d1["Age"].mean(), 1, 0)
d1['Bin_Plasma Storage Time']=np.where(d1['Plasma Storage Time']>d1['Plasma Storage Time'].mean(), 1, 0)

#Hard code 99% specificity cutoffs
cutoffs={'Random': 0.990594401, 'Site': 0.9862772124, 'Coll': 0.9935410899999999, 'Seq': 0.9906285177, 'Seq10': 0.9906285177, 'AltSite': 0.9922970853, 'AltColl': 0.9896170033999997, 'AltSeq': 0.9842252320000001, 'AltSeq10': 0.9842252320000001}

#Score and training/validation indicator columns for the various classifiers (splits)
splitCols={"Site_Score":'Site_Split',"CollDate_Score":"CollDate_Split","SeqSplit_Score":"Seq_Split","RandomSplit_Score":"Random_Split","AltSeqSplit_Score": "AltSeq_Split","AltCollDate_Score": "AltCollDate_Split","AltSite_Score": "AltSite_Split"}

#==== Unadjusted and adjusted OR calc
y='Outcome'
df_o= pd.DataFrame()
for Score in ["RandomSplit_Score","SeqSplit_Score","CollDate_Score","Site_Score","AltSeqSplit_Score","AltCollDate_Score","AltSite_Score"]:
   d_o=LR(d1,[Score],y)
   df_o = pd.concat([df_o, d_o], ignore_index=True)
   d_o=LR(d1,[Score,'Age', 'Gender', 'Tobacco', 'Plasma Storage Time','FC'],y)
   df_o = pd.concat([df_o, d_o], ignore_index=True)
df_o.to_csv("confounder_summary_LR.tsv",sep="\t")


#=== Weighted OR calc =========

#To isolate covariates, we will need to binarize them and equalize all other covariate di
def binarize(c):
    m={'Age':"Bin_Age",'Tobacco':"Tobacco", 'Gender':"Gender",'Plasma Storage Time':"Bin_Plasma Storage Time",'FC':"FC"}
    return(m[cov])

covariates = ['Tobacco','Age', 'Gender', 'Plasma Storage Time', 'FC']
balance_stats = []

for k in splitCols.keys():
 Score=k   
 for cov in covariates: #isolate each in turn
    y=binarize(cov) #use binary form of isolated cov
    c=[x for x in covariates if x!=cov] #all other covariates

    #fit a predictive model for binarized cov as a function of other covariates
    X=sm.add_constant(d1[c])
    ps_model = sm.Logit(d1[y], X).fit(disp=False)
    d1['ps'] = ps_model.predict(X) #now predict to get an estimate of the binarized cov

    #this is the definition of stabilized inverse probability weights
    p_exposed = d1[y].mean()
    d1['weights'] = np.where(
        d1[y]==1,
        p_exposed / d1['ps'],
        (1 - p_exposed) / (1 - d1['ps'])
    )

    #we are not capping the weights
    #cap = d1['weights'].quantile(0.995)
    #d1['weights'] = d1['weights'].clip(upper=cap)

    #now we use the weighted samples to predict outcome as a function of the covariates and the score
    #this gives weighted odds ratios and pvalues
    X=sm.add_constant(d1[covariates+[Score]])
    glm_model = sm.GLM(d1['Outcome'], X,family=sm.families.Binomial(),freq_weights=d1['weights']).fit()
    wOR= np.exp(glm_model.params)[Score]
    wse = glm_model.bse[Score]
    wp = glm_model.pvalues[Score]

    #repeat the same with un weighted samples to predict outcome as a function of the covariates and the score
    #this gives unweighted odds ratios and pvalues
    glm_model = sm.GLM(d1['Outcome'], X,family=sm.families.Binomial(),freq_weights=None).fit()
    uOR= np.exp(glm_model.params)[Score]
    use = glm_model.bse[Score]
    up = glm_model.pvalues[Score]        

    #now check for balance
    #are the smds close to 0 between the two groups cov=1 and cov=0 for all others covariates
    #smd stands for standardized mean difference
    #difference in weighted means between the two groups divided by the pooled weighted std dev
    for cov1 in covariates:
        if cov1!=cov:
            smd_unweighted = compute_smd(d1, cov1, y, weight_col=None)
            smd_weighted   = compute_smd(d1, cov1, y, weight_col='weights')
            balance_stats.append({
                'score': Score,
                'exposure': cov,
                'uOR':uOR,
                'wOR':wOR,                
                'uORCI':[uOR-1.96*use,uOR+1.96*use], #parametric CIs
                'wORCI':[wOR-1.96*wse,wOR+1.96*wse], #parametric CIs
                'uPVal': up,
                'wPVal': wp,
                'covariate': cov1,
                'SMD_unweighted': smd_unweighted,
                'SMD_weighted': smd_weighted
            })
        

balance_df = pd.DataFrame(balance_stats)
print(cov,"------------------------------\n")
print("\n",balance_df,"\n\n\n")
balance_df.to_csv("All_balanceORs.tsv",sep="\t")    
        
