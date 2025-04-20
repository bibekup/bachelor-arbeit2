# cnn_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import time
import psutil
import os
import GPUtil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the FER-2013 dataset test
def load_data(file_path):
    data = pd.read_csv(file_path)
    width, height = 48, 48
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(width, height))
    images = np.stack(pixels, axis=0)
    images = np.expand_dims(images, -1)  # Ensure shape (48, 48, 1)
    labels = to_categorical(data['emotion'])
    return images, labels

# Function to print resource usage
def print_resource_usage(label=""):
    print(f"\n Resource Usage {label}")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RAM usage: {mem_info.rss / (1024 ** 2):.2f} MB")
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"\nGPU Name: {gpu.name}")
            print(f"GPU Memory Used: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")
            print(f"GPU Load: {gpu.load * 100:.1f}%")
            print(f"GPU Temp: {gpu.temperature} °C")
    except:
        print("No GPU found or GPUtil not available.")

# Load data
file_path = 'fer2013.csv'
images, labels = load_data(file_path)
images = images / 255.0

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build CNN Model
model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')  # Output layer
    ])

    # Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train Model
print("\n[INFO] Training CNN Model...")
start_cnn = time.time()
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=60,
    verbose=1
)
end_cnn = time.time()
cnn_duration = end_cnn - start_cnn
print(f"\n CNN Training Time: {cnn_duration / 60:.2f} minutes")
print_resource_usage("[After CNN Training]")

# Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save CNN test data
#np.save('X_test_cnn.npy', X_test)
#np.save('y_test_cnn.npy', y_test)

# Save Model
#model.save('emotion_detection_model.h5')
print("[SAVED] CNN -> 'emotion_detection_model.h5'")
