
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("homeowners_synthetic_25vars.csv")

print("Dataset Shape:", df.shape)
print("Skewness:", df["total_loss"].skew())

plt.figure(figsize=(8,5))
sns.histplot(df["total_loss"], bins=50)
plt.title("Total Loss Distribution")
plt.tight_layout()
plt.savefig(f"{output_dir}/01_total_loss_distribution.png")
plt.close()


plt.figure(figsize=(8,5))
sns.histplot(np.log(df["total_loss"]), bins=50)
plt.title("Log Total Loss Distribution")
plt.tight_layout()
plt.savefig(f"{output_dir}/02_log_total_loss_distribution.png")
plt.close()


corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{output_dir}/03_correlation_heatmap.png")
plt.close()

print("\nCorrelation with Total Loss:")
print(corr["total_loss"].sort_values(ascending=False))


tier1_vars = [
    "roof_age",
    "roof_vulnerability",
    "dwelling_construction",
    "water_loss_recency",
    "prior_claims_5yr",
    "catastrophe_zone_score",
    "replacement_cost_index",
    "fire_protection_score",
    "structural_condition_index",
    "exposure_density_index"
]

for var in tier1_vars:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[var], y=df["total_loss"], alpha=0.2)
    plt.title(f"{var} vs Total Loss")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tier1_{var}.png")
    plt.close()


tier2_vars = [
    "insurance_lapses",
    "maintenance_quality_index",
    "liability_risk_index",
    "occupancy_risk_score",
    "neighborhood_stability_score",
    "deductible_level"
]

for var in tier2_vars:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[var], y=df["total_loss"], alpha=0.2)
    plt.title(f"{var} vs Total Loss")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tier2_{var}.png")
    plt.close()


tier3_vars = [
    "roof_wildfire_interaction",
    "water_canopy_interaction",
    "slope_burn_interaction",
    "replacement_exposure_interaction",
    "liability_occupancy_interaction",
    "age_protection_interaction"
]

for var in tier3_vars:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[var], y=df["total_loss"], alpha=0.2)
    plt.title(f"{var} vs Total Loss")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tier3_{var}.png")
    plt.close()
