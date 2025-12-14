import os
# Suppress TensorFlow oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# ------------------------------------------------------------
# 1️⃣ Load the label‑id ↔ animal‑name mapping
# ------------------------------------------------------------
LABEL_CSV = Path("../dataset/AnimalDataLabel.csv")   # adjust if you move the script
id_to_name: dict[int, str] = {}

with LABEL_CSV.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # CSV stores IDs as strings; convert to int for safety
        id_to_name[int(row["label_id"])] = row["animal_name"]

# ------------------------------------------------------------
# 2️⃣ Build an ordered list of class names
# ------------------------------------------------------------
# ImageDataGenerator expects the list to be ordered by the integer label.
# If any IDs are missing, we’ll fill the gap with a placeholder.
# Build an ordered list of class names matching the folder names.
# The CSV uses IDs starting at 1, so we skip the placeholder for 0.
max_id = max(id_to_name.keys())
# Keras expects folder names as strings; use the numeric IDs as folder names
class_names = [str(i) for i in range(1, max_id + 1)]

print("✅ Loaded class mapping (first 10 entries):")
for i in range(1, min(11, max_id + 1)):
    print(f"  {i}: {id_to_name.get(i, 'unknown')}" )

# ------------------------------------------------------------
# 3️⃣ Create the data generators
# ------------------------------------------------------------
TRAIN_DATA_PATH = Path("../dataset/repaired_image_train")
TEST_DATA_PATH  = Path("../dataset/repaired_image_test")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.0,          # we already split the data on disk
)

# `class_mode="categorical"` gives you one‑hot vectors that match the 99 classes.
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH,    
    target_size=(224, 224),        # same size you used in the preprocessing script
    batch_size=16,
    class_mode="categorical",
    shuffle=True,
    classes=class_names,           # <-- tells Keras the exact order of classes
)

test_generator = train_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    shuffle=False,
    classes=class_names,
)

# ------------------------------------------------------------
# 4️⃣ Build / compile / train the model (example)
# ------------------------------------------------------------
# Load the pre-trained MobileNetV2 model
# include_top=False removes the final classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model to prevent updating pre-trained weights during the first phase
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# Save the trained model
time_str = time.strftime("%Y%m%d-%H%M%S")
model_filename = f"animalClassifier_{time_str}.h5"
model.save(f"../prediction-model/{model_filename}")

# ------------------------------------------------------------
# 5️⃣ Save Model Metadata
# ------------------------------------------------------------
# Get Metrics
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

# Create comma-separated list of labels in order (0, 1, 2...)
# class_names are strings "1", "2"... 
# We need to map them back to names using id_to_name
ordered_labels = []
for cid in class_names:
    ordered_labels.append(id_to_name.get(int(cid), "unknown"))
output_labels_str = ",".join(ordered_labels)

# Restore missing model_filename definition
model_filename = f"animalClassifier_{time_str}.keras" 
csv_file_path = "../prediction-model/aaa-model-details.csv"

# Columns: model_name,prediction_type,no_of_output_categories,input_shape,
#          validation_accuracy,train_accuracy,model_path,model_type,
#          architecture,optimizer,learning_rate,epochs,output_labels

row_data = [
    f"animalClassifier_{time_str}",           # model_name
    "Categorical",                            # prediction_type
    len(class_names),                         # no_of_output_categories
    "(224,224,3)",                            # input_shape (Must match model input)
    f"{val_acc:.2f}",                         # validation_accuracy
    f"{train_acc:.2f}",                       # train_accuracy
    f"../prediction-model/{model_filename}",  # model_path
    "MobileNetV2",                            # model_type
    "MobileNetV2-GlobalAvgBase-Dense-256",    # architecture
    "Adam",                                   # optimizer
    "1e-4",                                   # learning_rate
    30,                                       # epochs
    output_labels_str                         # output_labels
]

# Append to CSV
try:
    file_exists = os.path.isfile(csv_file_path)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_name","prediction_type","no_of_output_categories","input_shape",
                "validation_accuracy","train_accuracy","model_path","model_type",
                "architecture","optimizer","learning_rate","epochs","output_labels"
            ])
        writer.writerow(row_data)
    print(f"Model metadata saved to {csv_file_path}")
except Exception as e:
    print(f"Error saving to CSV: {e}")
