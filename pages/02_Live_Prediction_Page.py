# pages/02_Live_Prediction_Page.py
import streamlit as st
import numpy as np
import tempfile
import os

from PIL import Image

from core.sidebar import load_sidebar
from core.model import load_model, predict
from core.preprocess import load_and_preprocess_image

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
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            input_image = camera_image

    with tab2:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            input_image = uploaded_file

    if input_image is not None:
        # Display image
        image = Image.open(input_image)
        st.image(image, caption="Input Image", width=300)

        if st.button("Predict"):
            with st.spinner("Processing..."):
                try:
                    # Preprocess
                    # Note: We need a way to pass the file content or save it temp.
                    # preprocess.py likely takes a path or handles bytes.
                    # Let's check preprocess.py first or adapt. 
                    # For now, I'll assume load_and_preprocess_image handles paths.
                    # I'll save to a temp path for compatibility.
                    
                    
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                         image.save(tmp.name)
                         tmp_path = tmp.name

                    # Prepare for model
                    # Fetch metadata for dynamic input_shape
                    from core.model import get_model_metadata
                    
                    target_size = (224, 224) # Default
                    
                    try:
                        meta_df = get_model_metadata()
                        current_filename = model_path.name
                        model_row = meta_df[meta_df['model_path'].apply(lambda x: str(x).endswith(current_filename))]
                        
                        if not model_row.empty:
                            shape_str = model_row.iloc[0]['input_shape']
                            clean_shape = shape_str.replace('(', '').replace(')', '').replace('"', '').replace("'", "")
                            dims = [int(x.strip()) for x in clean_shape.split(',')]
                            if len(dims) >= 2:
                                target_size = (dims[0], dims[1])
                    except Exception as meta_err:
                        print(f"Metadata fetch warning: {meta_err}")

                    processed_img = load_and_preprocess_image(tmp_path, target_size=target_size)
                    
                    # Cleanup
                    os.remove(tmp_path)

                    # Predict
                    predictions = predict(model, processed_img)
                    
                    # Parse results (Assuming classification)
                    # We might need class names. For now, showing raw top index.
                    class_idx = np.argmax(predictions, axis=1)[0]
                    confidence = np.max(predictions)
                    
                    st.markdown(f"### Prediction Result")
                    st.success(f"**Class Index:** {class_idx}")
                    st.info(f"**Confidence:** {confidence:.2f}")
                    
                except Exception as e:
                     st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    run()