import os
import csv
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image

PREDICTION_HISTORY_DIR = Path("prediction-history")

def save_prediction(model_name: str, image: Image.Image, predicted_label: str, confidence: float, class_index: int):
    """
    Saves the prediction image and logs the details to a CSV file.
    
    Args:
        model_name: Name of the model used.
        image: PIL Image object of the predicted image.
        predicted_label: Human-readable label of the prediction.
        confidence: Confidence score of the prediction.
        class_index: Numeric index of the predicted class.
    """
    # 1. Setup Directories
    model_dir = PREDICTION_HISTORY_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate Unique Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    image_filename = f"{timestamp}_{unique_id}.png"
    image_path = model_dir / image_filename
    
    # 3. Save Image
    try:
        image.save(image_path)
    except Exception as e:
        print(f"Error saving history image: {e}")
        return

    # 4. Save to CSV
    csv_path = model_dir / "prediction-history.csv"
    file_exists = csv_path.exists()
    
    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(["Timestamp", "Image Filename", "Predicted Label", "Confidence", "Class Index", "Satisfaction"])
            
            # Write record
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image_filename,
                predicted_label,
                f"{confidence:.4f}",
                class_index,
                0 # Default Satisfaction (0: Neutral, 1: Like, -1: Dislike)
            ])
            
    except Exception as e:
        print(f"Error saving history log: {e}")

def update_last_prediction_feedback(model_name: str, satisfaction_score: int):
    """
    Updates the Satisfaction column of the last entry in the model's history CSV.
    
    Args:
        model_name: Name of the model.
        satisfaction_score: 1 (Like), -1 (Dislike), or 0 (Neutral).
    """
    model_dir = PREDICTION_HISTORY_DIR / model_name
    csv_path = model_dir / "prediction-history.csv"
    
    if not csv_path.exists():
        print(f"History file not found for {model_name}")
        return

    try:
        # Read all rows
        rows = []
        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) > 1: # Ensure there's data besides header
            # Update last row's last column (Sustainability)
            # Check if header has 'Satisfaction'
            header = rows[0]
            if "Satisfaction" not in header:
                # Add column if missing (logic for migration if needed, but assuming new structure)
                header.append("Satisfaction")
                # Add 0 to all previous rows
                for i in range(1, len(rows)-1):
                    rows[i].append(0)
                # Update last row
                rows[-1].append(satisfaction_score)
            else:
                # Assuming Satisfaction is the last column
                idx_sat = header.index("Satisfaction")
                # Ensure row has enough columns
                if len(rows[-1]) <= idx_sat:
                    rows[-1].append(satisfaction_score)
                else:
                    rows[-1][idx_sat] = satisfaction_score
            
            # Write back
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
    except Exception as e:
        print(f"Error updating feedback: {e}")

