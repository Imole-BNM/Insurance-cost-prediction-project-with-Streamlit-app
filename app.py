"""Streamlit app: estimate annual medical insurance charges."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "insurance_model.joblib"

st.set_page_config(page_title="Insurance Cost Estimator", page_icon="🏥")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def build_features(age, sex, bmi, children, smoker, region, columns):
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "is_obese": int(bmi >= 30),
        "sex_male": int(sex == "male"),
        "smoker_yes": int(smoker == "yes"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
    }
    row["obese_smoker"] = row["is_obese"] * row["smoker_yes"]
    return pd.DataFrame([row])[columns]


st.title("🏥 Medical Insurance Cost Estimator")
st.caption("Predicts yearly insurance charges from the Kaggle 'Medical Cost Personal' dataset. "
           "For learning purposes only, not a real quote.")

bundle = load_model()
if bundle is None:
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.number_input("BMI", 15.0, 55.0, 26.0, step=0.1)
    children = st.number_input("Children", 0, 5, 0)
with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Estimate charges", type="primary"):
    X = build_features(age, sex, bmi, children, smoker, region, bundle["columns"])
    pred = max(float(bundle["model"].predict(X)[0]), 0.0)
    st.metric("Estimated yearly charges", f"${pred:,.0f}")
    if smoker == "yes":
        st.info("Smoking status is the strongest driver of cost in this dataset.")
