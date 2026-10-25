
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "printqueue_model.joblib"

st.set_page_config(
    page_title="PrintQueue",
    page_icon="🖨️",
    layout="centered",
)

st.title("🖨️ PrintQueue")
st.subheader("Photocopy & Printing Completion-Time Predictor")

st.write(
    "Estimate how long a printing job may take to be ready for collection."
)

st.info(
    "Academic prototype using synthetic data. Predictions are estimates, "
    "not guaranteed completion times."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("job_details"):
    st.subheader("Enter Job Details")

    page_count = st.number_input(
        "Pages in this job",
        min_value=1,
        max_value=5000,
        value=20,
        step=1,
    )

    jobs_ahead = st.number_input(
        "Number of jobs ahead",
        min_value=0,
        max_value=100,
        value=2,
        step=1,
    )

    pages_ahead = st.number_input(
        "Pages waiting ahead in the queue",
        min_value=0,
        max_value=20000,
        value=50,
        step=1,
    )

    printing_type = st.selectbox(
        "Printing type",
        ["BW", "Colour"],
    )

    binding_required = st.selectbox(
        "Binding required?",
        ["No", "Yes"],
    )

    available_printers = st.number_input(
        "Printers currently available",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )

    submitted = st.form_submit_button("Predict Completion Time")

if submitted:
    input_data = pd.DataFrame(
        [
            {
                "page_count": page_count,
                "jobs_ahead": jobs_ahead,
                "available_printers": available_printers,
                "pages_ahead": pages_ahead,
                "printing_type": printing_type,
                "binding_required": binding_required,
            }
        ]
    )

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, prediction)

    st.success(
        f"Estimated completion time: {prediction:.1f} minutes"
    )

    st.caption(
        "This estimate depends on the patterns learned from the training "
        "dataset. Actual time may differ due to workload, printer speed, "
        "paper jams, and finishing requirements."
    )



