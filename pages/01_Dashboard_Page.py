# pages/01_Dashboard_Page.py
import streamlit as st
from PIL import Image

from core.sidebar import (
    load_sidebar
)
from core.model import (
    load_model
)
from core.history import (
    update_last_prediction_feedback
)
from core.prediction import (
    run_prediction_pipeline
)

# --------------------------------------------------------------------------
# Page title
# --------------------------------------------------------------------------
st.title("📊 Dashboard")
st.write("Welcome to the unified dashboard. Select a model from sidebar and start predicting!")
st.markdown("---")

# --------------------------------------------------------------------------
# Page content
# --------------------------------------------------------------------------
with st.expander("How this app works?", expanded=False):
    """
        Steps:
        -----
        1. Load model from Sidebar
        2. Load the model logic
        3. Row 1: Image Selection & Preview
        4. Row 2: Prediction
        5. Row 3: Feedback
    """
def run():
    # 1. Load model from Sidebar
    # --------------
    model_path = load_sidebar()
    if not model_path:
        st.info("Please select a model from the sidebar to continue.")
        return

    # 2. Load the model logic
    # --------------
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # 3. Row 1: Image Selection & Preview
    # --------------
    r1_col1, r1_col2 = st.columns(2, border=True)
    
    input_image = None

    with r1_col1:
        st.subheader("Select Image")
        input_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    with r1_col2:
        st.subheader("Preview")
        if input_image:
            image = Image.open(input_image)
            st.image(image, caption="Uploaded Image", width=200)
        else:
            st.info("No image selected.")

    st.markdown("---")

    # 4. Row 2: Prediction
    # --------------
    if input_image is not None:
        # Generate a unique key for the current image to detect changes
        # Using filename and size as a proxy for identity
        current_image_key = f"{input_image.name}_{input_image.size}"
        
        # Initialize session state for this page if needed
        if 'last_image_key' not in st.session_state:
            st.session_state.last_image_key = None
            
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
            
        # Check if image changed
        if st.session_state.last_image_key != current_image_key:
            st.session_state.prediction_result = None
            st.session_state.last_image_key = current_image_key
            
        # 4.2: Predict button
        # --------------
        if st.button("Predict 🚀", type="primary", width="stretch"):
            st.session_state.prediction_result = None # Clear old
            with st.spinner("Processing..."):
                try:
                    
                    # Run the generalized pipeline
                    result = run_prediction_pipeline(
                        model=model,
                        model_path=model_path,
                        image=Image.open(input_image)
                    )
                    
                    # Store in session state
                    st.session_state.prediction_result = result
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        
        # 4.3: Display Result if exists in state
        # --------------
        if st.session_state.get('prediction_result'):
            res = st.session_state.prediction_result
            
            st.subheader("Prediction Results")
            c1, c2 = st.columns(2, border=True)
            c1.metric("Prediction", res['label'].title())
            c2.metric("Confidence", f"{res['confidence']*100:.2f}%")
            
            # 4.3.1: Feedback Section
            # --------------
            if not res['feedback_given']:
                # Center the feedback header
                st.markdown("<h3 style='text-align: center;'>Are you satisfied with this results?</h3>", unsafe_allow_html=True)
                col_f1, col_f2, col_f3, col_f4 = st.columns([2.5,1,1,2])
                with col_f2:
                    if st.button("👍", help="Like"):
                        update_last_prediction_feedback(res['model_name'], 1)
                        res['feedback_given'] = True
                        st.session_state.prediction_result = res # update state
                        st.rerun()
                with col_f3:
                    if st.button("👎", help="Dislike"):
                        update_last_prediction_feedback(res['model_name'], -1)
                        res['feedback_given'] = True
                        st.session_state.prediction_result = res
                        st.rerun()
            else:
                st.success("Thanks for your feedback!") 


if __name__ == "__main__":
    run()
