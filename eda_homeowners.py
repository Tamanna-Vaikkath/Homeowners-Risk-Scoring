
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.getcwd()

EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")
SHAP_DIR = os.path.join(BASE_DIR, "outputs", "shap")
INTER_DIR = os.path.join(BASE_DIR, "outputs", "interactions")

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(INTER_DIR, exist_ok=True)

print("Output folders created.")



df = pd.read_csv("enterprise_homeowners_synthetic_25vars.csv")
TARGET = "total_loss"

print("Dataset Loaded:", df.shape)

# TARGET DISTRIBUTION

plt.figure(figsize=(8,5))
sns.histplot(df[TARGET], bins=60, kde=True)
plt.title("Total Loss Distribution")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "target_distribution.png"))
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(np.log(df[TARGET]), bins=60, kde=True)
plt.title("Log Total Loss Distribution")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "log_target_distribution.png"))
plt.close()

# CORRELATION HEATMAP

corr = df.corr(numeric_only=True)

plt.figure(figsize=(14,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"))
plt.close()


corr[TARGET].sort_values(ascending=False).to_csv(
    os.path.join(EDA_DIR, "target_correlations.csv")
)



# TRAIN TEST SPLIT

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TWEEDIE MODEL

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tweedie = TweedieRegressor(power=1.5, alpha=0.01, max_iter=500)
tweedie.fit(X_train_scaled, y_train)

pred_t = tweedie.predict(X_test_scaled)

tweedie_metrics = pd.DataFrame({
    "Metric": ["R2", "MAE", "RMSE"],
    "Value": [
        r2_score(y_test, pred_t),
        mean_absolute_error(y_test, pred_t),
        np.sqrt(mean_squared_error(y_test, pred_t))
    ]
})

tweedie_metrics.to_csv(os.path.join(EDA_DIR, "tweedie_metrics.csv"), index=False)

# GBM MODEL

gbm = GradientBoostingRegressor(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

gbm.fit(X_train, y_train)

pred_g = gbm.predict(X_test)

gbm_metrics = pd.DataFrame({
    "Metric": ["R2", "MAE", "RMSE"],
    "Value": [
        r2_score(y_test, pred_g),
        mean_absolute_error(y_test, pred_g),
        np.sqrt(mean_squared_error(y_test, pred_g))
    ]
})

gbm_metrics.to_csv(os.path.join(EDA_DIR, "gbm_metrics.csv"), index=False)

# SHAP ANALYSIS

explainer = shap.Explainer(gbm, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, "shap_summary.png"))
plt.close()

# SHAP Importance Table
shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean |SHAP|": np.abs(shap_values.values).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False)

shap_importance.to_csv(
    os.path.join(SHAP_DIR, "shap_feature_importance.csv"),
    index=False
)

# SHAP INTERACTION HEATMAP

interaction_values = shap.TreeExplainer(gbm).shap_interaction_values(X_test)
interaction_strength = np.abs(interaction_values).mean(axis=0)

interaction_df = pd.DataFrame(
    interaction_strength,
    index=X.columns,
    columns=X.columns
)

plt.figure(figsize=(12,10))
sns.heatmap(interaction_df, cmap="viridis")
plt.title("SHAP Interaction Strength")
plt.tight_layout()
plt.savefig(os.path.join(INTER_DIR, "shap_interaction_heatmap.png"))
plt.close()

interaction_df.sum().sort_values(ascending=False).to_csv(
    os.path.join(INTER_DIR, "interaction_ranking.csv")
)

print("All EDA and SHAP outputs saved successfully.")