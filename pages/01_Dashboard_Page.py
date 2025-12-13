# pages/01_Dashboard_Page.py
import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
                   
from core.history import save_prediction        
from core.model import get_model_metadata
from core.sidebar import load_sidebar
from core.model import load_model, predict
from core.preprocess import load_and_preprocess_image
from core.model import get_model_metadata, get_predicted_label


st.title("📊 Dashboard")
st.write("Welcome to the unified dashboard. Select a model from sidebar and start predicting!")
st.markdown("---")
def run():
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

    # 3. Row 1: Image Selection & Preview
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
    if input_image is not None:
        # Check if we have a new image to clear previous state
        # Or simply key the state by file name?
        # For simplicity, if button pressed, overwrite state.
        
        # Initialize session state for this page if needed
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
            
        if st.button("Predict 🚀", type="primary", width="stretch"):
            st.session_state.prediction_result = None # Clear old
            with st.spinner("Processing..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        input_image.seek(0)
                        img_temp = Image.open(input_image)
                        img_temp.save(tmp.name)
                        tmp_path = tmp.name

                    # fetch metadata logic (kept same)
                    # ... [snipped for brevity, can reuse existing imports] 
                    # Assuming we can just run the logic block again or encapsulate.
                    # Since I am replacing the block, I must rewrite the logic.
                    
                    # Prepare for model
                    target_size = (224, 224) 
                    pred_type = "Categorical"
                    
                     # Fetch metadata
                    try:
                        meta_df = get_model_metadata()
                        current_filename = model_path.name
                        model_row = meta_df[meta_df['model_path'].apply(lambda x: str(x).endswith(current_filename))]
                        if not model_row.empty:
                            shape_str = model_row.iloc[0]['input_shape']
                            clean_shape = shape_str.replace('(', '').replace(')', '').replace('"', '').replace("'", "")
                            dims = [int(x.strip()) for x in clean_shape.split(',')]
                            if len(dims) >= 2: target_size = (dims[0], dims[1])
                            if 'prediction_type' in model_row.columns: pred_type = model_row.iloc[0]['prediction_type']
                    except: pass

                    processed_img = load_and_preprocess_image(tmp_path, target_size=target_size)
                    os.remove(tmp_path)
                    
                    predictions = predict(model, processed_img)
                    
                    # Parse ...
                    class_idx = 0
                    confidence = 0.0
                    
                    if pred_type == "Binary":
                         if predictions.shape[-1] == 1:
                            conf = predictions[0][0]
                            is_positive = conf > 0.5
                            class_idx = 1 if is_positive else 0
                            confidence = conf if is_positive else (1 - conf)
                         else:
                            class_idx = np.argmax(predictions, axis=1)[0]
                            confidence = np.max(predictions)
                    else:
                        class_idx = np.argmax(predictions, axis=1)[0]
                        confidence = np.max(predictions)
                    
                    label_name = get_predicted_label(model_path.name, class_idx)
                    
                    # Save History (Initial 0 satisfaction)
                    save_prediction(
                        model_name=model_path.name,
                        image=Image.open(input_image),
                        predicted_label=label_name,
                        confidence=confidence,
                        class_index=class_idx
                    )
                    
                    # Store in session state
                    st.session_state.prediction_result = {
                        "label": label_name,
                        "confidence": confidence,
                        "model_name": model_path.name,
                        "feedback_given": False
                    }
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        
        # Display Result if exists in state
        if st.session_state.get('prediction_result'):
            res = st.session_state.prediction_result
            
            st.subheader("Prediction Results")
            c1, c2 = st.columns(2, border=True)
            c1.metric("Prediction", res['label'].title())
            c2.metric("Confidence", f"{res['confidence']*100:.2f}%")
            
            # Feedback Section
            if not res['feedback_given']:
                st.write("Are you satisfied with this results?")
                col_f1, col_f2, col_f3 = st.columns([1,1,5])
                with col_f1:
                    if st.button("👍", help="Like"):
                        from core.history import update_last_prediction_feedback
                        update_last_prediction_feedback(res['model_name'], 1)
                        res['feedback_given'] = True
                        st.session_state.prediction_result = res # update state
                        st.rerun()
                with col_f2:
                    if st.button("👎", help="Dislike"):
                        from core.history import update_last_prediction_feedback
                        update_last_prediction_feedback(res['model_name'], -1)
                        res['feedback_given'] = True
                        st.session_state.prediction_result = res
                        st.rerun()
            else:
                st.success("Thanks for your feedback!") 


if __name__ == "__main__":
    run()
