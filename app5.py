import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


st.set_page_config(
    page_title="Homeowners Risk Intelligence Platform",
    layout="wide",
)


st.markdown("""
<style>
.main-title {font-size:32px;font-weight:700;}

.centered-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 40px;
}

.metric-card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 8px 30px rgba(0,0,0,0.06);
    text-align:center;
}
.section-card {
    background:white;
    padding:30px;
    border-radius:18px;
    box-shadow:0 10px 35px rgba(0,0,0,0.07);
    margin-top:25px;
}
.stButton>button {
    border-radius:10px;
    padding:8px 20px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("homeowners_synthetic_dataset.csv")

df = load_data()
TARGET = "annual_loss"

X_full = df.drop(columns=[TARGET])
y = df[TARGET]
portfolio_avg = y.mean()

numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
numeric_cols.remove(TARGET)

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()


@st.cache_resource
def train_model():

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=4,
        random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_full, y)
    return pipe

model = train_model()


if "page" not in st.session_state:
    st.session_state.page = 1

if "input_data" not in st.session_state:
    st.session_state.input_data = {}


# PAGE 1 — INPUT

if st.session_state.page == 1:

    st.markdown(
        '<div class="centered-title">Homeowners Risk Intelligence Platform</div>',
        unsafe_allow_html=True
    )
    

    st.markdown('<div class="main-title">Policy Risk Input</div><br>', unsafe_allow_html=True)

    input_data = st.session_state.input_data

    col1, col2 = st.columns(2)

    with col1:
        input_data["home_age"] = st.slider("Home Age (Years)", 0, 100, input_data.get("home_age",20))
        input_data["roof_age"] = st.slider("Roof Age (Years)", 0, 50, input_data.get("roof_age",10))
        input_data["square_footage"] = st.slider("Living Area (Sq Ft)", 500, 8000, input_data.get("square_footage",2500))
        input_data["coverage_a"] = st.slider("Coverage A ($)", 50000, 2000000, input_data.get("coverage_a",300000), step=50000)
        input_data["property_value"] = st.slider("Property Value ($)", 50000, 2000000, input_data.get("property_value",300000), step=50000)
        input_data["iso_class"] = st.slider("ISO Fire Protection Class (1–10)", 1, 10, input_data.get("iso_class",5))

    with col2:
        input_data["fire_station_distance"] = st.slider("Distance to Fire Station (Miles)", 0.1, 20.0, input_data.get("fire_station_distance",3.0))
        input_data["credit_score"] = st.slider("Credit Score (300–850)", 300, 850, input_data.get("credit_score",700))
        input_data["prior_claims_5yr"] = st.slider("Claims in Past 5 Years", 0, 10, input_data.get("prior_claims_5yr",0))
        input_data["water_loss_recency"] = st.slider("Months Since Last Water Loss", 0, 60, input_data.get("water_loss_recency",12))
        input_data["wind_hail_score"] = st.slider("Wind/Hail Risk Score (0–100)", 0, 100, input_data.get("wind_hail_score",40))
        input_data["crime_index"] = st.slider("Crime Index (0–100)", 0, 100, input_data.get("crime_index",30))
        input_data["deductible"] = st.selectbox("Deductible ($)", [500,1000,1500,2500,5000],
                                               index=[500,1000,1500,2500,5000].index(input_data.get("deductible",1000)))

    for col in categorical_cols:
        default = input_data.get(col, df[col].mode()[0])
        input_data[col] = st.selectbox(col.replace("_"," ").title(),
                                       df[col].unique(),
                                       index=list(df[col].unique()).index(default))

    st.session_state.input_data = input_data

    if st.button("Next"):
        st.session_state.page = 2
        st.rerun()


# PAGE 2 — TIER OVERVIEW

elif st.session_state.page == 2:

    st.markdown('<div class="main-title">Risk Tier Architecture</div><br>', unsafe_allow_html=True)

    tier_table = pd.DataFrame({
        "Tier": ["Tier 1 – Structural","Tier 2 – Behavioral","Tier 3 – Geographic"],
        "Variables": [
            "Home Age, Roof Age, Coverage A, Property Value, ISO Class",
            "Claims History, Water Loss Recency, Credit Score, Deductible",
            "Crime Index, Wind/Hail Score, Wildfire Zone, Coastal Exposure"
        ],
        "Pricing Impact": [
            "Drives claim severity and replacement volatility",
            "Impacts frequency and behavioral risk",
            "Captures catastrophe and correlated exposure"
        ]
    })

    st.dataframe(tier_table, use_container_width=True)

    col1, col2 = st.columns(2)
    if col1.button("Back"):
        st.session_state.page = 1
        st.rerun()
    if col2.button("Next"):
        st.session_state.page = 3
        st.rerun()


# PAGE 3 — DASHBOARD

elif st.session_state.page == 3:

    st.markdown('<div class="main-title">Risk Intelligence Dashboard</div><br>', unsafe_allow_html=True)

    input_df = pd.DataFrame([st.session_state.input_data])

    for col in X_full.columns:
        if col not in input_df.columns:
            if col in numeric_cols:
                input_df[col] = df[col].median()
            else:
                input_df[col] = df[col].mode()[0]

    input_df = input_df[X_full.columns]

    pred = float(model.predict(input_df)[0])
    margin = 0.30
    premium = pred * (1 + margin)

    p10, p90 = np.percentile(y,10), np.percentile(y,90)
    score = np.clip(round(100*(pred-p10)/(p90-p10),1),0,100)
    segment = "Low Risk" if score < 35 else "Moderate Risk" if score < 70 else "High Risk"

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h3>${pred:,.0f}</h3><div>Expected Loss</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>${premium:,.0f}</h3><div>Recommended Premium</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>{score}</h3><div>Risk Score</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h3>{segment}</h3><div>Risk Segment</div></div>', unsafe_allow_html=True)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=["Policy"], y=[pred], marker_color="#2563eb"))
    fig_bar.add_trace(go.Bar(x=["Portfolio Avg"], y=[portfolio_avg], marker_color="#f59e0b"))
    fig_bar.update_layout(template="plotly_white", barmode="group")
    st.plotly_chart(fig_bar, use_container_width=True)

    processed = model.named_steps["preprocessor"].transform(input_df)
    explainer = shap.Explainer(model.named_steps["model"])
    shap_values = explainer(processed)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values.values[0]
    }).sort_values(by="Impact", key=abs, ascending=False)

    fig_shap = px.bar(shap_df.head(10),
                      x="Impact",
                      y="Feature",
                      orientation="h",
                      color="Impact",
                      color_continuous_scale="RdBu",
                      title="Top Drivers of Risk")
    st.plotly_chart(fig_shap, use_container_width=True)

    radar_vals = []
    for col in numeric_cols:
        minv,maxv=df[col].min(),df[col].max()
        radar_vals.append(100*(input_df[col].values[0]-minv)/(maxv-minv))

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_vals,
        theta=numeric_cols,
        fill='toself',
        line=dict(color="#10b981")
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0,100])),
        template="plotly_white"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    ratio = pred / portfolio_avg
    top_driver = shap_df.iloc[0]["Feature"]
    top_3 = shap_df.head(3)["Feature"].tolist()
    variance = pred - portfolio_avg

    st.markdown(f"""
### Underwriting Analysis

**Portfolio Context**
- Expected loss: **${pred:,.0f}**
- This represents **{round(ratio*100,1)}%** of portfolio average.
- Risk score **{score}/100** places this policy in the **{segment}** tier of the distribution.

**SHAP Interpretation**
- Primary loss driver: **{top_driver}**
- Secondary contributors: **{top_3[1]}**, **{top_3[2]}**

**Pricing Adequacy**
- Technical premium including 30% margin: **${premium:,.0f}**
- Margin accounts for fixed expenses, volatility load, and required return on capital.
- Premium is directly linked to modeled loss expectation rather than static rate tables.

**Underwriting Action Framework**
- Evaluate mitigation opportunities impacting the top driver.
- Review claims history for frequency clustering vs isolated loss.
- Consider deductible optimization to improve risk-adjusted performance.
- For elevated segments, assess catastrophe aggregation exposure and reinsurance impact.
""")

    if segment == "Low Risk":
        st.success("Risk profile is stable and below portfolio average. Approval at indicated pricing is supported.")
    elif segment == "Moderate Risk":
        st.info("Moderate exposure profile. Align with technical pricing and review mitigation adjustments.")
    else:
        st.error("Elevated modeled risk. Detailed underwriting review recommended prior to binding.")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Back"):
        st.session_state.page = 2
        st.rerun()