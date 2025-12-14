import csv
import os
from datetime import datetime

# Centralized path for model details
CSV_FILE_PATH = "../prediction-model/aaa-model-details.csv"

def save_model_metadata(
    model_name: str,
    prediction_type: str,
    no_of_output_categories: int,
    input_shape: str,
    val_acc: float,
    train_acc: float,
    model_path: str,
    model_type: str,
    architecture: str,
    optimizer: str,
    learning_rate: str,
    epochs: int,
    output_labels: str
):
    """
    Appends model metadata to the CSV file.
    """
    
    # Columns: model_name,prediction_type,no_of_output_categories,input_shape,
    #          validation_accuracy,train_accuracy,model_path,model_type,
    #          architecture,optimizer,learning_rate,epochs,output_labels

    row_data = [
        model_name,
        prediction_type,
        no_of_output_categories,
        input_shape,
        f"{val_acc:.2f}",
        f"{train_acc:.2f}",
        model_path,
        model_type,
        architecture,
        optimizer,
        learning_rate,
        epochs,
        output_labels
    ]

    file_exists = os.path.isfile(CSV_FILE_PATH)
    
    try:
        # Create directory if it doesn't exist (though presumably it does)
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "model_name","prediction_type","no_of_output_categories","input_shape",
                    "validation_accuracy","train_accuracy","model_path","model_type",
                    "architecture","optimizer","learning_rate","epochs","output_labels"
                ])
            writer.writerow(row_data)
        print(f"✅ Model metadata saved to {CSV_FILE_PATH}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")
