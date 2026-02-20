import numpy as np
import pandas as pd

np.random.seed(42)
n = 15000

# Tier 1 
roof_age = np.random.normal(0, 1, n)
dwelling_construction = np.random.normal(0, 1, n)
roof_vulnerability = np.random.normal(0, 1, n)
prior_claims_5yr = np.random.poisson(1.5, n)
water_loss_recency = np.random.normal(0, 1, n)


# Tier 2 
catastrophe_zone_score = np.random.normal(0, 1, n)
replacement_cost_index = np.random.normal(0, 1, n)
fire_protection_score = np.random.normal(0, 1, n)
structural_condition_index = np.random.normal(0, 1, n)
exposure_density_index = np.random.normal(0, 1, n)
insurance_lapses = np.random.poisson(0.5, n)


# Tier 3 
roof_wildfire_interaction = roof_vulnerability * catastrophe_zone_score
water_canopy_interaction = water_loss_recency * exposure_density_index
slope_burn_interaction = catastrophe_zone_score * roof_age
replacement_exposure_interaction = replacement_cost_index * exposure_density_index
age_protection_interaction = roof_age * fire_protection_score
liability_risk_index = np.random.normal(0, 1, n)
maintenance_quality_index = np.random.normal(0, 1, n)
neighborhood_stability_score = np.random.normal(0, 1, n)
deductible_level = np.random.normal(0, 1, n)
occupancy_risk_score = np.random.normal(0, 1, n)
swimming_pool = np.random.binomial(1, 0.3, n)
fire_sprinklers = np.random.binomial(1, 0.4, n)
monitored_alarm = np.random.binomial(1, 0.5, n)
liability_occupancy_interaction = liability_risk_index * occupancy_risk_score



loss = (
    30000
    + 9000 * roof_age
    + 11000 * roof_vulnerability
    + 10000 * dwelling_construction
    + 15000 * np.sqrt(prior_claims_5yr + 1)
    + 8000 * water_loss_recency
)


loss += 20000 * (roof_age > 1.0)
loss += 25000 * (catastrophe_zone_score > 1.2)

# Strong interaction amplification 
loss += 15000 * (roof_wildfire_interaction ** 2)

# Log-based nonlinear scaling
loss += 10000 * np.log1p(np.abs(replacement_exposure_interaction))

# Secondary nonlinearities
loss += 8000 * (slope_burn_interaction ** 2)
loss += 7000 * np.abs(water_canopy_interaction)


loss += 12000 * (liability_occupancy_interaction ** 2)

loss -= 9000 * monitored_alarm
loss -= 12000 * fire_sprinklers

loss += np.random.gamma(shape=2, scale=7000, size=n)

loss = np.maximum(loss, 1000)


df = pd.DataFrame({
    "roof_age": roof_age,
    "dwelling_construction": dwelling_construction,
    "roof_vulnerability": roof_vulnerability,
    "prior_claims_5yr": prior_claims_5yr,
    "water_loss_recency": water_loss_recency,
    "catastrophe_zone_score": catastrophe_zone_score,
    "replacement_cost_index": replacement_cost_index,
    "fire_protection_score": fire_protection_score,
    "structural_condition_index": structural_condition_index,
    "exposure_density_index": exposure_density_index,
    "insurance_lapses": insurance_lapses,
    "roof_wildfire_interaction": roof_wildfire_interaction,
    "water_canopy_interaction": water_canopy_interaction,
    "slope_burn_interaction": slope_burn_interaction,
    "replacement_exposure_interaction": replacement_exposure_interaction,
    "age_protection_interaction": age_protection_interaction,
    "liability_risk_index": liability_risk_index,
    "maintenance_quality_index": maintenance_quality_index,
    "neighborhood_stability_score": neighborhood_stability_score,
    "deductible_level": deductible_level,
    "occupancy_risk_score": occupancy_risk_score,
    "swimming_pool": swimming_pool,
    "fire_sprinklers": fire_sprinklers,
    "monitored_alarm": monitored_alarm,
    "liability_occupancy_interaction": liability_occupancy_interaction,
    "total_loss": loss
})

df.to_csv("homeowners_synthetic_25vars_v2.csv", index=False)

print("Dataset Generated Successfully!")
