
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("homeowners_synthetic_25vars_v2.csv")


X = df.drop(columns=["total_loss"])
y = df["total_loss"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Performance")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

df["predicted_loss"] = model.predict(X)

min_loss = df["predicted_loss"].min()
max_loss = df["predicted_loss"].max()

df["risk_score"] = 100 * (
    (df["predicted_loss"] - min_loss) /
    (max_loss - min_loss)
)

print("\nSample Risk Scores:")
print(df[["predicted_loss", "risk_score"]].head())

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance)

print("\nGenerating SHAP summary plot...")

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

os.makedirs("model_outputs", exist_ok=True)

shap.summary_plot(shap_values, X_test, show=False)
import matplotlib.pyplot as plt
plt.savefig("model_outputs/shap_summary.png")
plt.close()

print("SHAP summary plot saved to model_outputs folder.")


os.makedirs("models", exist_ok=True)
model.save_model("models/xgboost_model.json")

print("\nModel saved successfully.")
