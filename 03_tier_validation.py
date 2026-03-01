
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")



DATA_PATH = "homeowners_synthetic_dataset.csv"
TARGET = "annual_loss"

TIER_1 = [
    "prior_claims_5yr",
    "home_age",
    "wind_hail_score",
    "roof_age"
]

TIER_2 = [
    "credit_score",
    "crime_index",
    "fire_station_distance"
]


df = pd.read_csv(DATA_PATH)

X_t1 = df[TIER_1]
X_t12 = df[TIER_1 + TIER_2]
y = df[TARGET]


# TRAIN / TEST SPLIT

X1_train, X1_test, y_train, y_test = train_test_split(
    X_t1, y, test_size=0.30, random_state=42
)

X12_train, X12_test, _, _ = train_test_split(
    X_t12, y, test_size=0.30, random_state=42
)


# FIT TWEEDIE MODEL

model_t1 = TweedieRegressor(
    power=1.5,
    alpha=0.01,
    link="log",
    max_iter=1000
)

model_t12 = TweedieRegressor(
    power=1.5,
    alpha=0.01,
    link="log",
    max_iter=1000
)

model_t1.fit(X1_train, y_train)
model_t12.fit(X12_train, y_train)


# PREDICTIONS

pred_t1 = model_t1.predict(X1_test)
pred_t12 = model_t12.predict(X12_test)


# EVALUATION METRICS

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Mean_Predicted": np.mean(y_pred)
    }

results_t1 = evaluate(y_test, pred_t1)
results_t12 = evaluate(y_test, pred_t12)


print("\n================================================")
print("TIER VALIDATION RESULTS")
print("================================================")

print("\nTIER 1 MODEL")
for k, v in results_t1.items():
    print(f"{k}: {round(v,4)}")

print("\nTIER 1 + TIER 2 MODEL")
for k, v in results_t12.items():
    print(f"{k}: {round(v,4)}")

incremental_r2 = results_t12["R2"] - results_t1["R2"]

print("\nINCREMENTAL R2 LIFT:", round(incremental_r2,6))

print("================================================")


# COEFFICIENT REVIEW

coef_df = pd.DataFrame({
    "Variable": TIER_1 + TIER_2,
    "Coefficient": list(model_t12.coef_)
})

print("\nMODEL COEFFICIENTS (Tier 1 + Tier 2)")
print(coef_df.sort_values("Coefficient", ascending=False))