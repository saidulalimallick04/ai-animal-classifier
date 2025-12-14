# ----------------------------------------
import os
# Suppress TensorFlow oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import random
import csv
import keras
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D, Conv2D

# Start Execution Timer
start_time = time.time()
print(f"⏱️ Script started at: {time.ctime(start_time)}")

# -------------------------------------
# 1. Path to the training and testing data
# -------------------------------------
TRAIN_DATA_PATH = "../dataset/binary/repaired_images_train"
TEST_DATA_PATH = "../dataset/binary/repaired_images_test"

# -------------------------------------
# 2. Display a random image from the training data
# -------------------------------------
# Function wrapped to avoid polluting global namespace or just keep as inline script
try:
    category = random.choice(['cats', 'dogs'])
    category_path = os.path.join(TRAIN_DATA_PATH, category)
    if os.path.exists(category_path):
        image_files = os.listdir(category_path)
        if image_files:
            random_image = random.choice(image_files)
            image_path = os.path.join(category_path, random_image)
            image = Image.open(image_path)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Category: {category}")
            # plt.show() # Commented out to prevent blocking in non-interactive runs
except Exception as e:
    print(f"⚠️ Warning: Could not display random image: {e}")

# -------------------------------------
# 3. Data Augmentation and Data Generator
# -------------------------------------
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH,
    target_size=(150, 150),
    class_mode="binary"
)
val_generator = val_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(150, 150),
    class_mode="binary"
)

# -------------------------------------
# 4. Model Building (SimpleCNN)
# -------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation="sigmoid")
])

# -------------------------------------
# 5. Model Compilation
# -------------------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------------
# 6. Model Training
# -------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# -------------------------------------
# 7. Model Saving
# -------------------------------------
time_str = time.strftime("%Y%m%d-%H%M%S")
model_filename = f'binary_model_{time_str}.keras'
model_dir = '../Model/binary_models'
os.makedirs(model_dir, exist_ok=True)
model.save(f'{model_dir}/{model_filename}')

# -------------------------------------
# 8. Model Evaluation
# -------------------------------------
model.evaluate(val_generator)

# -------------------------------------
# 9. Save Metadata to CSV
# -------------------------------------
# Calculate final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

csv_file_path = "../prediction-model/aaa-model-details.csv"

# Columns: model_name, prediction_type, no_of_output_categories, input_shape,
#          validation_accuracy, train_accuracy, model_path, model_type,
#          architecture, optimizer, learning_rate, epochs, output_labels

row_data = [
    f"binary_model_{time_str}",                 # model_name
    "Binary",                                   # prediction_type
    2,                                          # no_of_output_categories
    "(150,150,3)",                              # input_shape
    f"{final_val_acc:.2f}",                     # validation_accuracy
    f"{final_train_acc:.2f}",                   # train_accuracy
    f"../Model/binary_models/{model_filename}", # model_path
    "SimpleCNN",                                # model_type
    "Conv2D-32-64-128-Dense-512",               # architecture
    "Adam",                                     # optimizer
    "default",                                  # learning_rate
    10,                                         # epochs
    "cats,dogs"                                 # output_labels
]

try:
    file_exists = os.path.isfile(csv_file_path)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model_name", "prediction_type", "no_of_output_categories", "input_shape",
                "validation_accuracy", "train_accuracy", "model_path", "model_type",
                "architecture", "optimizer", "learning_rate", "epochs", "output_labels"
            ])
        writer.writerow(row_data)
    print(f"✅ Model metadata saved to {csv_file_path}")
except Exception as e:
    print(f"❌ Error saving to CSV: {e}")

# Rate the Duration
end_time = time.time()
duration = end_time - start_time
print(f"⏱️ Script ended at: {time.ctime(end_time)}")
print(f"⏳ Total execution time: {duration:.2f} seconds")
