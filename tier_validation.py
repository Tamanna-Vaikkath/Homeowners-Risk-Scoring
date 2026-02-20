import pandas as pd
import numpy as np

df = pd.read_csv("homeowners_synthetic_25vars.csv")

tier1_vars = [
    'roof_age',
    'dwelling_construction',
    'roof_vulnerability',
    'prior_claims_5yr',
    'water_loss_recency'
]

tier2_vars = [
    'catastrophe_zone_score',
    'replacement_cost_index',
    'fire_protection_score',
    'structural_condition_index',
    'exposure_density_index',
    'insurance_lapses'
]

def validate_signal(variable):
    df['bin'] = pd.qcut(df[variable], 10, duplicates='drop')
    grouped = df.groupby('bin')['total_loss'].mean().reset_index()
    print("\nVariable:", variable)
    print(grouped)

print("Tier 1 Validation")

for var in tier1_vars:
    validate_signal(var)

print("\nTier 2 Validation")

for var in tier2_vars:
    validate_signal(var)
