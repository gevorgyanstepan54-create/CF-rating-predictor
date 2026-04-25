import streamlit as st
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_models():
    return {
        "div1": joblib.load("cf_model_div1_1000plus.joblib"),
        "div1_div2": joblib.load("cf_model_div1_div2_1000plus.joblib"),
        "div2_parallel": joblib.load("cf_model_div2_parallel_1000plus.joblib"),
        "div2_solo": joblib.load("cf_model_div2_solo_1000plus.joblib"),
        "div3": joblib.load("cf_model_div3_1000plus.joblib"),
    }

models = load_models()

def predict_all(contest_type, old_rating, percentile):
    # Predict Delta
    row = pd.DataFrame({"old_rating": [old_rating], "percentile": [percentile]})
    delta = float(models[contest_type].predict(row)[0])
    
    # Estimate Performance
    ratings_grid = np.arange(1000, 4001)
    grid_df = pd.DataFrame({
        "old_rating": ratings_grid,
        "percentile": np.full(len(ratings_grid), percentile, dtype=float)
    })
    preds = models[contest_type].predict(grid_df)
    perf = int(ratings_grid[np.argmin(np.abs(preds))])
    
    return delta, old_rating + delta, perf

# STreamlit
st.set_page_config(page_title="CF Rating Predictor")
st.title("Codeforces Rating Predictor")

with st.form("prediction_form"):
    contest = st.selectbox("Contest Type", list(models.keys()))
    rating = st.number_input("Your Current Rating", value=1500, step=1)
    percent = st.number_input("Percentile (0.0=Top, 1.0=Bottom)",value=0.5)
    submit = st.form_submit_button("Predict Result")

if submit:
    delta, new_r, performance = predict_all(contest, rating, percent)
    
    cols = st.columns(3)
    cols[0].metric("Rating Change", f"{delta:+.1f}")
    cols[1].metric("New Rating", int(new_r))
    cols[2].metric("Performance", performance)