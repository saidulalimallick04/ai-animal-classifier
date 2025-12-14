"""Sidebar utilities for the core package.

Provides a ``load_sidebar`` function that renders a Streamlit sidebar
allowing the user to select a TensorFlow/Keras model from the
``prediction-model`` directory. Only files with ``.h5`` or ``.keras``
extensions are shown.
"""
import streamlit as st
from pathlib import Path

def load_sidebar() -> Path | None:
    """Render a model-selection sidebar.

    Returns
    -------
    Path | None
        The full path to the selected model file, or ``None`` if no model
        is found/selected.
    """
    st.sidebar.title("Model Selector")
    model_dir = Path("prediction-model")
    if not model_dir.is_dir():
        st.sidebar.error("Model directory not found.")
        return None

    # Gather model files with allowed extensions
    allowed_ext = {".h5", ".keras"}
    model_files = [f for f in model_dir.iterdir() if f.is_file() and f.suffix in allowed_ext]
    if not model_files:
        st.sidebar.info("No model files (.h5/.keras) found in the prediction‑model folder.")
        return None

    model_names = [f.name for f in model_files]
    choice = st.sidebar.selectbox("Select a model", model_names)
    selected_path = model_dir / choice
    st.sidebar.success(f"✅ Selected model: {choice}")
    return selected_path


def sidebar_info() -> None:
    '''
    Render a sidebar with information about the application.
    '''
    pass