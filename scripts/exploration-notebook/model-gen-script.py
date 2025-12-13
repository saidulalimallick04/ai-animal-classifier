import os
import random
import shutil
import zipfile
import numpy as np

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten

from tensorflow.keras.layers import Conv2D,MaxPool2D
import matplotlib.pyplot as plt
from PIL import Image

TRAIN_DATA_PATH="../dataset/repaired_images_train"
TEST_DATA_PATH="../dataset/repaired_images_test"


category= random.choice(['cats','dogs'])
category_path=os.path.join(train_dir,category)

image_files=os.listdir(category_path)

random_image=random.choice(image_files)

image_path=os.path.join(category_path,random_image)

image=Image.open(image_path)

plt.imshow(image)
plt.axis('off')
plt.title(f"Category: {category}")




train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=20,horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generater=train_datagen.flow_from_directory(train_dir,target_size=(150,150),class_mode="binary")
val_generater=val_datagen.flow_from_directory(val_dir,target_size=(150,150),class_mode="binary")


model=keras.Sequential([
    keras.layers.Conv2D(32,(3,3), activation='relu',input_shape=(150,150,3)),
    keras.layers.MaxPool2D(2,2),

    
    keras.layers.Conv2D(64,(3,3), activation='relu'),
    keras.layers.MaxPool2D(2,2),

    
    keras.layers.Conv2D(128,(3,3), activation='relu'),
    keras.layers.MaxPool2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(1,activation="sigmoid")
    
])


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(train_generater,validation_data=val_generater,epochs=10)


model.save('../Model/cat&dogClassifier02.h5')

model.evaluate()