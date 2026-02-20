import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

features = tier1_vars + tier2_vars

X = df[features]
y = df['total_loss']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TweedieRegressor(power=1.5, alpha=0.01, link='log')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Tweedie Baseline Performance")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coefficients.sort_values(by="Coefficient", ascending=False))
