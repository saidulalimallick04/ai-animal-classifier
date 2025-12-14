"""Sidebar utilities for the core package.

Provides a ``load_sidebar`` function that renders a Streamlit sidebar
allowing the user to select a TensorFlow/Keras model from the
``prediction-model`` directory. Only files with ``.h5`` or ``.keras``
extensions are shown.
"""
import streamlit as st
from pathlib import Path
from core.model import get_model_metadata


def load_sidebar() -> Path | None:
    """Render a model-selection sidebar.

    Returns
    -------
    Path | None
        The full path to the selected model file, or ``None`` if no model
        is found/selected.
    """
    st.sidebar.title("Model Selector")
    # Load metadata from CSV
    try:
        df = get_model_metadata()
    except ImportError:
        # Fallback if core.model import fails (unlikely)
        st.sidebar.error("Could not import model metadata utility.")
        return None
    if df.empty:
        st.sidebar.warning("No models found in the registry (JSON).")
        return None

    # Sort models by creation/name if possible, here just by name
    model_names = df.get('model_name').tolist()
    
    
    # Allow user to select by model Name
    choice = st.sidebar.selectbox("Select a model", model_names)
    
    # Retrieve the path for the selected model
    row = df[df.get('model_name') == choice].iloc[0]
    raw_path = str(row.get('model_path'))
    
    # FIX: The CSV contains paths relative to the 'scripts/' folder (e.g. "../prediction-model/...")
    # We are running from the root, so we need to adjust them.
    # 1. replace "../prediction-model" with "prediction-model"
    # 2. replace "../Model" with "Model"
    # 3. If it's just a filename, assume it's in prediction-model (legacy)

    # FIX: The CSV contains paths relative to the 'scripts/' folder (e.g. "../prediction-model/...")
    # We are running from the root, so we need to adjust them.
    clean_path = raw_path
    if clean_path.startswith("../"):
        clean_path = clean_path.replace("../", "", 1)
    full_path = Path(clean_path)
    
    # Check existence
    if not full_path.exists():
        # Check for G-Drive ID if file is missing
        g_id = row.get('g_drive_file_id', 'no-url-found')
        
        if g_id and g_id != 'no-url-found':
            # Attempt Download
            st.sidebar.warning(f"⚠️ Model missing locally. Downloading from Google Drive...")
            try:
                import gdown
                with st.spinner("Downloading model... Please wait..."):
                    # Ensure directory exists (though full_path includes filename)
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    url = f'https://drive.google.com/uc?id={g_id}'
                    output = str(full_path)
                    gdown.download(url, output, quiet=False)
                    
                if full_path.exists():
                     st.sidebar.success("✅ Download complete!")
                     st.rerun()
                else:
                     st.sidebar.error("❌ Download failed. File not found after attempt.")
                     return None
            except ImportError:
                 st.sidebar.error("❌ 'gdown' library not installed. Cannot download model.")
                 return None
            except Exception as e:
                 st.sidebar.error(f"❌ Download failed: {e}")
                 return None
        else:
            st.sidebar.error(f"⚠️ Model file missing at: {full_path}")
            st.sidebar.error("ℹ️ No Google Drive ID found in metadata to auto-download.")
            return None
        
    st.sidebar.success(f"✅ Selected: {choice}")
    return full_path


def sidebar_info() -> None:
    '''
    Render a sidebar with information about the application.
    '''
    pass