import os
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path

from core.model import get_model_metadata, predict, get_predicted_label
from core.preprocess import load_and_preprocess_image
from core.history import save_prediction

def run_prediction_pipeline(model, model_path: Path, image: Image.Image) -> dict:
    """
    Runs the full prediction pipeline:
    1. Preprocesses the image (using metadata for shape).
    2. Runs inference.
    3. Decodes the output (Binary vs Categorical).
    4. Saves the result to history.
    5. Returns the result dictionary.
    """
    
    # 1. Fetch Metadata for shape and type
    target_size = (224, 224) 
    pred_type = "Categorical"
    
    try:
        meta_df = get_model_metadata()
        current_filename = model_path.name
        # Flexible matching
        model_row = meta_df[meta_df['model_path'].apply(lambda x: str(x).endswith(current_filename))]
        
        if not model_row.empty:
            # Parse Input Shape
            shape_str = model_row.iloc[0]['input_shape']
            clean_shape = shape_str.replace('(', '').replace(')', '').replace('"', '').replace("'", "")
            dims = [int(x.strip()) for x in clean_shape.split(',')]
            if len(dims) >= 2: 
                target_size = (dims[0], dims[1])
            
            # Parse Prediction Type
            if 'prediction_type' in model_row.columns: 
                pred_type = model_row.iloc[0]['prediction_type']
    except Exception as e:
        print(f"Metadata fetch warning: {e}")

    # 2. Save Image to Temp (for preprocessing function that expects path)
    # TODO: Refactor load_and_preprocess_image to accept PIL object directly to avoid IO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        processed_img = load_and_preprocess_image(Path(tmp_path), target_size=target_size)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 3. Predict
    predictions = predict(model, processed_img)

    # 4. Decode Prediction
    class_idx = 0
    confidence = 0.0

    if pred_type == "Binary":
         if predictions.shape[-1] == 1:
            conf = float(predictions[0][0])
            is_positive = conf > 0.5
            class_idx = 1 if is_positive else 0
            confidence = conf if is_positive else (1 - conf)
         else:
            class_idx = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions))
    else:
        # Categorical
        class_idx = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
    
    label_name = get_predicted_label(model_path.name, class_idx)
    
    # 5. Save History
    save_prediction(
        model_name=model_path.name,
        image=image,
        predicted_label=label_name,
        confidence=confidence,
        class_index=class_idx
    )
    
    # 6. Return Result
    return {
        "label": label_name,
        "confidence": confidence,
        "model_name": model_path.name,
        "feedback_given": False,
        "prediction_type": pred_type
    }
