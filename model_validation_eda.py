
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("homeowners_synthetic_25vars_v2.csv")

os.makedirs("validation_outputs", exist_ok=True)


print("\nTARGET DISTRIBUTION")
print(df["total_loss"].describe())

plt.figure()
plt.hist(df["total_loss"], bins=50)
plt.title("Distribution of Total Loss")
plt.xlabel("Total Loss")
plt.ylabel("Frequency")
plt.savefig("validation_outputs/total_loss_distribution.png")
plt.close()


corr = df.corr(numeric_only=True)["total_loss"].sort_values(ascending=False)

print("\nCORRELATION WITH TOTAL LOSS")
print(corr)

corr.to_csv("validation_outputs/correlation_with_loss.csv")

# Tier 1

tier1_vars = [
    "roof_age",
    "roof_vulnerability",
    "prior_claims_5yr"
]

print("\nTIER 1 MONOTONICITY")

for var in tier1_vars:
    df["bin"] = pd.qcut(df[var], 10, duplicates="drop")
    grouped = df.groupby("bin")["total_loss"].mean().reset_index()
    
    print(f"\nVariable: {var}")
    print(grouped)

    plt.figure()
    plt.plot(grouped.index, grouped["total_loss"])
    plt.title(f"Monotonicity Check - {var}")
    plt.xlabel("Bin")
    plt.ylabel("Average Loss")
    plt.savefig(f"validation_outputs/monotonicity_{var}.png")
    plt.close()

print("\nVARIANCE CHECK")
print("Mean:", df["total_loss"].mean())
print("Std Dev:", df["total_loss"].std())
print("Max Loss:", df["total_loss"].max())
print("Min Loss:", df["total_loss"].min())

# Coefficient of Variation
cv = df["total_loss"].std() / df["total_loss"].mean()
print("Coefficient of Variation:", cv)

print("\nValidation EDA completed successfully.")
