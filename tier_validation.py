
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score
from scipy.stats import ks_2samp



BASE_DIR = os.getcwd()

VAL_DIR = os.path.join(BASE_DIR, "outputs", "validation")
os.makedirs(VAL_DIR, exist_ok=True)

print("Validation folder created.")



df = pd.read_csv("enterprise_homeowners_synthetic_25vars.csv")
TARGET = "total_loss"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



gbm = GradientBoostingRegressor(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

gbm.fit(X_train, y_train)

# MONOTONICITY VALIDATION

print("Running Monotonicity Validation...")

monotonic_features = [
    "roof_age",
    "catastrophe_zone_score",
    "prior_claims_5yr"
]

monotonic_results = []

for feature in monotonic_features:
    sorted_df = df.sort_values(feature)
    preds = gbm.predict(sorted_df.drop(columns=[TARGET]))
    correlation = np.corrcoef(sorted_df[feature], preds)[0,1]

    monotonic_results.append((feature, correlation))

monotonic_df = pd.DataFrame(
    monotonic_results,
    columns=["Feature", "Correlation_with_Prediction"]
)

monotonic_df.to_csv(os.path.join(VAL_DIR, "monotonicity_results.csv"), index=False)

# PARTIAL DEPENDENCE PLOTS

print("Generating PDP plots...")

pdp_features = [
    "roof_age",
    "catastrophe_zone_score",
    "prior_claims_5yr"
]

for feature in pdp_features:
    fig, ax = plt.subplots(figsize=(6,4))
    PartialDependenceDisplay.from_estimator(
        gbm, X_train, [feature], ax=ax
    )
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, f"pdp_{feature}.png"))
    plt.close()

# CROSS-VALIDATION STABILITY

print("Running Cross Validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    cv_scores.append(r2_score(y_val, preds))

cv_df = pd.DataFrame({
    "Fold": range(1,6),
    "R2": cv_scores
})

cv_df.to_csv(os.path.join(VAL_DIR, "cross_validation_results.csv"), index=False)


print("Checking Data Drift...")

drift_results = []

for col in X.columns:
    stat, p_value = ks_2samp(X_train[col], X_test[col])
    drift_results.append((col, p_value))

drift_df = pd.DataFrame(
    drift_results,
    columns=["Feature", "KS_p_value"]
)

drift_df.to_csv(os.path.join(VAL_DIR, "data_drift_results.csv"), index=False)

# RISK SEGMENTATION CUT-OFFS

print("Creating Risk Segments...")

df["predicted_loss"] = gbm.predict(X)

low_cut = df["predicted_loss"].quantile(0.33)
high_cut = df["predicted_loss"].quantile(0.66)

def risk_bucket(x):
    if x <= low_cut:
        return "Low Risk"
    elif x <= high_cut:
        return "Medium Risk"
    else:
        return "High Risk"

df["risk_segment"] = df["predicted_loss"].apply(risk_bucket)

segment_summary = df.groupby("risk_segment")["predicted_loss"].agg(
    ["count","mean","min","max"]
)

segment_summary.to_csv(os.path.join(VAL_DIR, "risk_segmentation_summary.csv"))

print("Enterprise Validation Completed Successfully.")