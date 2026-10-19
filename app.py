
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("predictive_maintenance_model_compressed.pkl")
    features = joblib.load("model_features.pkl")
    return model, features
model, features = load_model()

st.title("Self-Healing Machines")
st.subheader("AI-Powered Predictive Maintenance")
st.write("NASA C-MAPSS FD001 engine RUL prediction demo.")

uploaded_file = st.file_uploader("Upload test_FD001.txt", type=["txt"])

if uploaded_file:
    columns = (
        ["engine_id", "cycle"] +
        [f"setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}" for i in range(1, 22)]
    )

    data = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    data = data.dropna(axis=1, how="all")
    data.columns = columns

    idx = data.groupby("engine_id")["cycle"].idxmax()
    latest = data.loc[idx].sort_values("engine_id")

    predictions = np.maximum(
        model.predict(latest[features]), 0
    )

    results = pd.DataFrame({
        "Engine ID": latest["engine_id"].to_numpy(),
        "Current Cycle": latest["cycle"].to_numpy(),
        "Predicted RUL (cycles)": np.round(predictions, 1)
    })

    results["Risk Status"] = results[
        "Predicted RUL (cycles)"
    ].apply(lambda x: "High" if x <= 20 else
            "Medium" if x <= 50 else "Low")

    st.metric("Engines analyzed", len(results))
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download predictions",
        results.to_csv(index=False),
        "maintenance_predictions.csv",
        "text/csv"
    )

    st.caption(
        "Risk thresholds are illustrative demo rules. "
        "Predicted RUL is an estimate, not a guaranteed failure time."
    )
