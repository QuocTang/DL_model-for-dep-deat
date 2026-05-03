"""
Streamlit UI for single-image inference using YOLO deep features + trained ML model.

Run:
  c:/Users/ADMIN/Desktop/personal_train/ML_deppfeat/.venv/Scripts/python.exe -m streamlit run infer_ui_streamlit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from infer_utils import (
    DEFAULT_ML_MODEL,
    DEFAULT_RESULTS_JSON,
    DEFAULT_YOLO_WEIGHTS,
    DeepFeatureMLPredictor,
)


st.set_page_config(page_title="Durian Disease Inference", page_icon="leaf", layout="wide")
st.title("Durian Disease Inference UI")
st.caption("Upload an image to run inference with YOLO deep features + XGBoost model")


@st.cache_resource
def load_predictor(
    yolo_path: str,
    ml_model_path: str,
    results_json_path: str,
    layer_str: str,
    imgsz: int,
):
    layers = [int(x.strip()) for x in layer_str.split(",") if x.strip()]
    predictor = DeepFeatureMLPredictor(
        yolo_weights=Path(yolo_path),
        ml_model_path=Path(ml_model_path),
        results_json_path=Path(results_json_path),
        layers=layers,
        imgsz=imgsz,
    )
    return predictor


with st.sidebar:
    st.header("Model Paths")
    yolo_path = st.text_input("YOLO weights", value=str(DEFAULT_YOLO_WEIGHTS))
    ml_model_path = st.text_input("ML model (.joblib)", value=str(DEFAULT_ML_MODEL))
    results_json_path = st.text_input("results.json", value=str(DEFAULT_RESULTS_JSON))

    st.header("Inference Settings")
    layer_str = st.text_input("Feature layers", value="4,6,9")
    imgsz = st.number_input("Input size", min_value=320, max_value=1280, value=640, step=32)
    top_k = st.slider("Top-K classes", min_value=1, max_value=10, value=5)

    load_clicked = st.button("Load/Reload Models", type="primary")

if "predictor" not in st.session_state or load_clicked:
    try:
        st.session_state.predictor = load_predictor(
            yolo_path=yolo_path,
            ml_model_path=ml_model_path,
            results_json_path=results_json_path,
            layer_str=layer_str,
            imgsz=int(imgsz),
        )
        st.sidebar.success("Models loaded")
    except Exception as exc:
        st.sidebar.error(f"Load failed: {exc}")
        st.stop()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption=uploaded.name, use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            pred = st.session_state.predictor.predict_from_rgb(image_np, top_k=top_k)

        st.subheader("Prediction")
        st.write(f"Predicted class: **{pred['predicted_class']}**")

        top_df = pd.DataFrame(pred["top_k"])
        st.dataframe(top_df, use_container_width=True)

        chart_df = top_df[["class_name", "probability"]].set_index("class_name")
        st.bar_chart(chart_df)

        st.json(pred)
else:
    st.info("Upload an image to start inference.")
