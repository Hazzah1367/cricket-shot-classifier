"""Launch from project root: streamlit run app.py."""

import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from cricket_shot_classifier.predict import Predictor

st.set_page_config(page_title="Cricket Shot Classifier", page_icon="🏏")
st.title("🏏 Cricket Shot Classifier")
st.write("Upload one clear batting image: cover drive, pull shot or sweep shot.")
st.caption(
    "Experimental MVP. Probabilities are uncalibrated; other shots and non-cricket images "
    "are outside its scope. A single frame may be ambiguous."
)
checkpoint = Path(os.environ.get("CRICKET_CHECKPOINT", "artifacts/resnet18/best.pt"))


@st.cache_resource
def load_model(path, modified):
    torch.set_num_threads(4)
    return Predictor(path)


if not checkpoint.exists():
    st.info(
        "Train the model first using the README instructions, or extract the supplied checkpoint."
    )
    st.stop()
upload = st.file_uploader("Batting image", type=["jpg", "jpeg", "png", "webp"])
if upload is not None:
    try:
        image = Image.open(upload)
        st.image(image, width="stretch")
        predictor = load_model(str(checkpoint), checkpoint.stat().st_mtime_ns)
        result = predictor.predict(image)
        st.subheader(result["prediction"].replace("_", " ").title())
        st.dataframe(
            [
                {"Shot": key.replace("_", " ").title(), "Probability": value}
                for key, value in result["probabilities"].items()
            ],
            hide_index=True,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not process this image: {exc}")
