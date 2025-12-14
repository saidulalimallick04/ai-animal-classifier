# pages/02_Live_Prediction_Page.py
import streamlit as st
import numpy as np
import tempfile
import os

from PIL import Image

from core.sidebar import load_sidebar
from core.model import load_model, predict
from core.preprocess import load_and_preprocess_image
from core.prediction import run_prediction_pipeline
from core.history import update_last_prediction_feedback

def run():
    st.title("🎥 Live Prediction")

    # 1. Load model from Sidebar
    model_path = load_sidebar()
    if not model_path:
        st.info("👈 Please select a model from the sidebar to continue.")
        return

    # 2. Load the model logic
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # 3. Image Input: Camera or Upload
    tab1, tab2 = st.tabs(["📸 Camera", "📂 Upload"])
    
    input_image = None
    
    with tab1:
        col1, col2 = st.columns(2, border=True)
        with col1:
            camera_image = st.camera_input("Take a picture",)
            if camera_image:
                input_image = camera_image
        with col2:
            st.write("Captured Image")
            if camera_image:
                st.image(camera_image, caption="Input Image", width='stretch')
            
    with tab2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            input_image = uploaded_file
            st.image(uploaded_file, caption="Uploaded Image", width=300)

    if input_image is not None:
        # Load image (Preview handled in tabs)
        image = Image.open(input_image)
        
        # Generate a unique key for the current image
        # For CameraInput, it's bytes. For FileUploader, it's a file-like object with name.
        if hasattr(input_image, 'name'):
            current_image_key = f"{input_image.name}_{input_image.size}"
        else:
            # Camera input (bytesIO), use hash/size
            current_image_key = f"cam_{input_image.getvalue().__sizeof__()}"

        # Initialize session state for this page if needed
        if 'live_last_image_key' not in st.session_state:
            st.session_state.live_last_image_key = None
            
        if 'live_prediction_result' not in st.session_state:
            st.session_state.live_prediction_result = None
            
        # Check if image changed (Clear result if new image)
        if st.session_state.live_last_image_key != current_image_key:
            st.session_state.live_prediction_result = None
            st.session_state.live_last_image_key = current_image_key

        if st.button("Predict 🚀", type="primary", width='stretch'):
            st.session_state.live_prediction_result = None # Clear old
            with st.spinner("Processing..."):
                try:
                    # Run the generalized pipeline
                    result = run_prediction_pipeline(
                        model=model,
                        model_path=model_path,
                        image=image
                    )
                    
                    # Store in session state
                    st.session_state.live_prediction_result = result
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        # Display Result if exists
        if st.session_state.get('live_prediction_result'):
            res = st.session_state.live_prediction_result
            
            st.subheader("Prediction Results")
            c1, c2 = st.columns(2, border=True)
            c1.metric("Prediction", res['label'].title())
            c2.metric("Confidence", f"{res['confidence']*100:.2f}%")
            
            # Feedback Section
            if not res['feedback_given']:
                st.markdown("<h4 style='text-align: center;'>Satisfied?</h4>", unsafe_allow_html=True)
                col_f1, col_f2, col_f3, col_f4 = st.columns([2,1,1,2])
                with col_f2:
                    if st.button("👍", key="live_up"):
                        update_last_prediction_feedback(res['model_name'], 1)
                        res['feedback_given'] = True
                        st.session_state.live_prediction_result = res
                        st.rerun()
                with col_f3:
                    if st.button("👎", key="live_down"):
                        update_last_prediction_feedback(res['model_name'], -1)
                        res['feedback_given'] = True
                        st.session_state.live_prediction_result = res
                        st.rerun()
            else:
                st.success("Thanks for your feedback!")

if __name__ == "__main__":
    run()