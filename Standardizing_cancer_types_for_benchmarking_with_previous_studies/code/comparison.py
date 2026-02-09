from scipy import stats
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="statsmodels"
)


#generates n_boot diffs of stage wise standard sensitivity between ref_test and cur_test
#creates a dataframe where columns are stages
#rows are the diffs
def bootstrap_stage_diffs_multitest(
    df, ref_test, cur_test, n_boot=1000
):
    out = []

    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        seed = rng.integers(0, 1_000_000_000)
        boot = df.sample(frac=1, replace=True, random_state=seed)
        #print(boot)
        boot_model = linearmodel(boot,"total")

        sens = pd.concat(
            [standardized_sensitivity(boot_model, df, t) for t in [cur_test, ref_test]],
            ignore_index=True
        )
        #print(sens)

        
        wide = sens.pivot(
            index="stage",
            columns="test",
            values="Sensitivity"
        )

        diff = wide[cur_test] - wide[ref_test]

        out.append(diff)

    return pd.concat(out, axis=1).T

#get CIs for the standardized sensitivity for each test
def bootstrap_ci(df, n_boot=1000):

    boot_results = []

    rng = np.random.default_rng(42)
    for b in range(n_boot):
        seed = rng.integers(0, 1_000_000_000)
        boot_df = df.sample(frac=1, replace=True, random_state=seed)    

        #boot_df['prop'] = (boot_df['positive'] + 0.0005) / (boot_df['total']+0.01)
        model = linearmodel(boot_df,"total")

        for t in boot_df["test"].unique():
            tmp = standardized_sensitivity(model, df, t) #tmp has test, stage and sensitivity (stdized)
            tmp["boot"] = b #and now bootstrap iteration number
            boot_results.append(tmp)


    #stack iterations in a dataframe
    boot_df = pd.concat(boot_results)
    print(boot_df)

    #calculate standardized sensitivity CIs test and stagewise
    ci = (
        boot_df
        .groupby(["test", "stage"])["Sensitivity"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "CI_low", 0.975: "CI_high"})
    )

    return ci

#result: a glm model to predict detection rate as a
#function of stage, cancer_type, stage*cancer_type and test
#test_value: a given test name, e.g., ours or CancerSEEK or CCGA
#df: a dataframe containing stage, cancer_type, and total sample count
#the same stage, cancer_type combination may repeat on the rows
#either because of bootstrapping or because different rows
#arose from different tests
#this function will predict the detection rate for each row
#and then aggregate that over all the rows keeping the totals in each
#row in mind to obtained an overall weighted sensitivity by stage
def standardized_sensitivity(result, df, test_value):

    # Explicitly select only model variables
    tmp = df.copy()

    pred_df = tmp[[
        "stage",
        "cancer_type",
        "total"
    ]].copy()

    # Add test as a clean column, ignore actual test value
    pred_df["test"] = test_value

    # Predict probabilities
    pred_df["pred"] = result.predict(pred_df)
    
    # Stage-wise standardized sensitivity
    out = (
        pred_df
        .groupby("stage",observed=False)
        .apply(lambda x: np.average(x["pred"], weights=x["total"]),include_groups=False)
        .reset_index(name="Sensitivity")
    )

    out["test"] = test_value
    return out

def linearmodel(df,w):
    if w!=None: w=df[w]
    model = smf.glm(
        formula="prop ~ test * stage + cancer_type",    
        #formula="positive + failures ~ test * stage + cancer_type",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=w
    ).fit()
    return(model)



df = pd.read_csv("../InputFiles/Comparison-Klein.tsv", sep="\t")

df=df.rename(columns={"Study":"test"})
df=df.rename(columns={"Cancer Class":"cancer_type"})
df=df.rename(columns={"Clinical Stage":"stage"})
df=df.rename(columns={"Total":"total"})
df=df.rename(columns={"Test Positive":"positive"})
stage_order = ["I", "II", "III", "IV"]
df["stage"] = pd.Categorical(df["stage"], stage_order, ordered=True)
df["test"] = df["test"].astype("category")
df["cancer_type"] = df["cancer_type"].astype("category")
df["failures"] = df["total"] - df["positive"]
df['prop'] = (df['positive'] + 0.00000005) / (df['total']+0.0001)

All14=["Prostate","Kidney","Urinary Bladder","Lung","Esophagus","Stomach","Colorectal","Cervix","Breast","Uterus","Ovary","Liver","Pancreas","Gall Bladder"]
CCGAPreSpec9=["Colorectal", "Esophagus", "Kidney", "Liver", "Lung", "Ovary", "Pancreas", "Stomach", "Urinary Bladder"]
nonScreenable9=["Kidney","Urinary Bladder","Esophagus","Stomach","Uterus","Ovary","Liver","Pancreas","Gall Bladder"]
cSEEK8=["Lung","Esophagus","Stomach","Colorectal","Breast","Ovary","Liver","Pancreas"]
lists={"All14":All14,"CCGAPreSpec9":CCGAPreSpec9,"nonScreenable9":nonScreenable9,"cSEEK8":cSEEK8}


#for each set of cancer types
for k in lists.keys():
    l= lists[k]
    xdf=df[df["cancer_type"].isin(l)]
    

    #main fit, fits detection rates weighted by total #samples, as a function of stage, cancer type, stage*cancer type, and test
    result=linearmodel(xdf,"total")
    #print(result.summary())

    #take rows from all tests to compute the
    #stagewise sensitivity for each test
    #thus standardizing the comparison between tests
    tests = xdf["test"].cat.categories.tolist()
    sens_df = pd.concat(
        [standardized_sensitivity(result, xdf, t) for t in tests],
        ignore_index=True
    )


    #bootstrap to get ci and add those columns to sens_df
    ci_df = bootstrap_ci(xdf, n_boot=1000)
    final_df = sens_df.merge(ci_df, on=["test", "stage"])
    #print(final_df)
    final_df.to_csv("../Tables/"+k+" CIs.tsv",sep="\t")

    #now bootstrap to get n_boot differences in stagewise standardized sensitivity
    #these differences are stacked as rows, columns being the stages
    boot_diffs = bootstrap_stage_diffs_multitest(
        xdf,
        ref_test="CCGA",
        cur_test="This Study",
        n_boot=1000
    )

    #print(boot_diffs)
    #since diffs are cur - ref
    #null: cur<=ref
    #alternative: cur>ref
    #pvalue is the fraction of bootstrap iterations where diff is negative
    summary = boot_diffs.apply( #to each column, i.e., stage
        lambda x: pd.Series({
            "diff": x.mean(),
            "lcl": np.percentile(x, 2.5),
            "ucl": np.percentile(x, 97.5),
            "p": (x <= 0).mean()
            #"p": 2 * min((x <= 0).mean(), (x >= 0).mean())
        })
    )

    
    #print(summary)
    summary.to_csv("../Tables/"+k+" CCGA-pvalue.tsv",sep="\t")

    #repeat of above for CancerSEEK vs ThisStudy
    boot_diffs = bootstrap_stage_diffs_multitest(
        xdf,
        ref_test="CancerSEEK",
        cur_test="This Study",
        n_boot=1000
    )

    summary = boot_diffs.apply(
        lambda x: pd.Series({
            "diff": x.mean(),
            "lcl": np.percentile(x, 2.5),
            "ucl": np.percentile(x, 97.5),
            "p": (x <= 0).mean()
            #"p": 2 * min((x <= 0).mean(), (x >= 0).mean())
        })
    )

    #print(summary)
    summary.to_csv("../Tables/"+k+" CancerSEEK-pvalue.tsv",sep="\t")
