"""model module

Utilities for loading the TensorFlow/Keras model and making predictions.
"""
import tensorflow as tf
from pathlib import Path
import pandas as pd
import streamlit as st

def get_model_metadata() -> pd.DataFrame:
    """Read the model details CSV."""
    csv_path = Path("prediction-model/aaa-model-details.csv")
    if not csv_path.exists():
        st.error(f"Model details file not found at: {csv_path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Error reading model details: {e}")
        return pd.DataFrame()

def select_model(model_names: list[str]) -> str | None:
    """Render a selectbox for model selection."""
    if not model_names:
        st.warning("No models available to select.")
        return None
    return st.selectbox("Select a model to view details", model_names)


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a Keras model from the given path.
    Raises FileNotFoundError if the file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(str(model_path))

def predict(model: tf.keras.Model, image_array):
    """Run prediction on a pre‑processed image array.
    Returns the raw prediction tensor.
    """
    return model.predict(image_array)

def get_predicted_label(model_name: str, class_index: int) -> str:
    """Get the label for a given class index and model name from metadata."""
    try:
        df = get_model_metadata()
        if df.empty:
            return f"Unknown (No Metadata)"
        
        # Filter by model_name
        # The CSV has 'model_name' column (e.g. 'animalClassifier02')
        row = df[df['model_name'] == model_name]
        
        if row.empty:
            # Fallback: check if model_name is a path and matches filename
            # This handles cases where model_name passed is actually the filename
            row = df[df['model_path'].apply(lambda x: str(x).endswith(model_name) or Path(str(x)).name == model_name)]
        
        if row.empty:
            return f"Unknown Model ({model_name})"
            
        labels_str = row.iloc[0]['output_labels']
        pred_type = "Categorical"
        if 'prediction_type' in row.columns and not pd.isna(row.iloc[0]['prediction_type']):
            pred_type = row.iloc[0]['prediction_type']

        if pd.isna(labels_str) or not labels_str:
            return f"Class {class_index}"
            
        # Parse labels
        clean_labels = labels_str.replace('"', '').replace("'", "")
        labels = [x.strip() for x in clean_labels.split(',')]
        
        # Robust check based on type
        if pred_type == "Binary":
            if len(labels) != 2:
                return f"Error: Binary model must have 2 labels (found {len(labels)})"
            if class_index not in [0, 1]:
                 return f"Error: Binary class index must be 0 or 1 (got {class_index})"
        
        if 0 <= class_index < len(labels):
            return labels[class_index]
        else:
            return f"Unknown Class ({class_index})"
            
    except Exception as e:
        print(f"Error getting label: {e}")
        return f"Error ({class_index})"
