
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv("enterprise_homeowners_synthetic_25vars.csv")  

TARGET = "total_loss"  

y = df[TARGET]

print("Dataset Loaded.")
print("Shape:", df.shape)
print()



tier1_features = [
    "roof_age","replacement_cost_index","catastrophe_zone_score",
    "structural_condition_score","fire_protection_score",
    "construction_quality_score","elevation_risk_score",
    "wind_exposure_score","distance_to_coast","urban_density_score"
]

tier2_features = tier1_features + [
    "prior_claims_5yr","water_loss_recency_score",
    "occupancy_risk_score","liability_exposure_score",
    "maintenance_quality_score","inspection_findings_score",
    "neighborhood_stability_score","claim_frequency_area_score",
    "credit_proxy_score","local_inflation_factor"
]

tier3_features = tier2_features + [
    "roof_wildfire_interaction",
    "coast_wind_interaction",
    "age_condition_interaction",
    "claims_recency_interaction",
    "liability_occupancy_interaction"
]


X_train, X_test, y_train, y_test = train_test_split(
    df[tier3_features], y, test_size=0.2, random_state=42
)


# TIER 1 MODEL
print("Training Tier 1 Model...")

model_t1 = GradientBoostingRegressor(random_state=42)
model_t1.fit(X_train[tier1_features], y_train)

pred_t1 = model_t1.predict(X_test[tier1_features])
r2_t1 = r2_score(y_test, pred_t1)

print("Tier 1 R²:", round(r2_t1, 4))


# TIER 2 MODEL
print("\nTraining Tier 1 + Tier 2 Model...")

model_t2 = GradientBoostingRegressor(random_state=42)
model_t2.fit(X_train[tier2_features], y_train)

pred_t2 = model_t2.predict(X_test[tier2_features])
r2_t2 = r2_score(y_test, pred_t2)

tier2_lift = r2_t2 - r2_t1

print("Tier 1 + Tier 2 R²:", round(r2_t2, 4))
print("Tier 2 Incremental Lift:", round(tier2_lift, 4))



print("\nTraining Full Tier 1 + 2 + 3 Model...")

model_t3 = GradientBoostingRegressor(random_state=42)
model_t3.fit(X_train[tier3_features], y_train)

pred_t3 = model_t3.predict(X_test[tier3_features])
r2_t3 = r2_score(y_test, pred_t3)

tier3_lift = r2_t3 - r2_t2

mae = mean_absolute_error(y_test, pred_t3)
rmse = np.sqrt(mean_squared_error(y_test, pred_t3))

print("Full Model R²:", round(r2_t3, 4))
print("Tier 3 Incremental Lift:", round(tier3_lift, 4))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


joblib.dump(model_t3, "enterprise_gbm.pkl")
print("\nFinal enterprise_gbm.pkl saved.")

print("\n===================================================")
print("THREE-TIER ARCHITECTURE SUMMARY")
print("===================================================")
print("Tier 1 Variance Explained:", round(r2_t1*100,2), "%")
print("Tier 2 Additional Lift:", round(tier2_lift*100,2), "%")
print("Tier 3 Interaction Lift:", round(tier3_lift*100,2), "%")
print("Total Variance Explained:", round(r2_t3*100,2), "%")
print("===================================================")