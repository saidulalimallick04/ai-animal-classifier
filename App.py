import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image 
import gdown

# Google Drive file ID and destination
FILE_ID = '1QSnx8bCdhS2Y38x7CwciQqCJt6rKSumn'  # replace with your file ID
MODEL_PATH = 'model.h5'

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# Streamlit UI
st.title("Cat Dog Classifier ")

st.header("Upload An Image to Predict")

uploaded_file=st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])

if st.button("Predict"):
    if uploaded_file is not None:
        img=Image.open(uploaded_file)
        img=img.resize((150,150))

        img_array=image.img_to_array(img)/255.0

        img_array=np.expand_dims(img_array,axis=0)

        prediction=model.predict(img_array)

        result="Dog" if prediction[0][0]>0.5 else "Cat"


        st.success(f"Predicted Image : **{result}**")

        st.image(uploaded_file,caption="Uploaded Image",width="stretch")
        
        

    else: 
        st.error("Upload an Image")
