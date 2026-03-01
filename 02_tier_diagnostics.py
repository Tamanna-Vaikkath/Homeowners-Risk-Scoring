
import pandas as pd
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr


DATA_PATH = "homeowners_synthetic_dataset.csv"
TARGET = "annual_loss"
OUTPUT_DIR = "tier_diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)



df = pd.read_csv(DATA_PATH)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols.remove(TARGET)

print("="*60)
print("RUNNING TIER DIAGNOSTICS")
print("="*60)

# VIF ANALYSIS

print("\n================ VIF ANALYSIS ================")

X = df[numeric_cols]

vif_data = []

for i in range(len(X.columns)):
    vif_data.append({
        "Variable": X.columns[i],
        "VIF": variance_inflation_factor(X.values, i)
    })

vif_df = pd.DataFrame(vif_data)
vif_df = vif_df.sort_values("VIF", ascending=False)

vif_df.to_csv(f"{OUTPUT_DIR}/vif_table.csv", index=False)

print("\nTop 5 VIF Variables:")
print(vif_df.head(5))

# MONOTONIC STRENGTH 

print("\n================ MONOTONIC STRENGTH ================")

mono_results = []

for col in numeric_cols:
    corr, pval = spearmanr(df[col], df[TARGET])

    mono_results.append({
        "Variable": col,
        "Spearman_Corr": corr,
        "Abs_Corr": abs(corr),
        "P_Value": pval
    })

mono_df = pd.DataFrame(mono_results)
mono_df = mono_df.sort_values("Abs_Corr", ascending=False)

mono_df.to_csv(f"{OUTPUT_DIR}/monotonic_strength.csv", index=False)

print("\nTop 10 Variables by Monotonic Strength:")
print(mono_df.head(10))

# AUTO TIER CANDIDATE FLAGGING

print("\n================ AUTO TIER SUGGESTION ================")

tier_candidates = mono_df.copy()

tier_candidates["Tier_Suggestion"] = np.where(
    tier_candidates["Abs_Corr"] > 0.15,
    "Tier 1 Candidate",
    np.where(
        tier_candidates["Abs_Corr"] > 0.05,
        "Tier 2 Candidate",
        "Weak Signal"
    )
)

tier_candidates.to_csv(f"{OUTPUT_DIR}/tier_suggestions.csv", index=False)

print(tier_candidates.head(15))

print("\n================================================")
print("TIER DIAGNOSTICS COMPLETE")
print("Outputs saved in:", OUTPUT_DIR)
print("================================================")