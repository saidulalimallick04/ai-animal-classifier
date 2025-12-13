import csv
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    classes=class_names,           # <-- tells Keras the exact order of classes
)

test_generator = train_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    classes=class_names,
)

# ------------------------------------------------------------
# 4️⃣ Build / compile / train the model (example)
# ------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
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
model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# Save the trained model
model.save("../prediction-model/animalClassifier.h5")