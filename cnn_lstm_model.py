# cnn_lstm_model.py
import numpy as np
import pandas as pd
import time
import os
import psutil
import GPUtil
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Bidirectional)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


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

# Load FER-2013
def load_data(file_path):
    data = pd.read_csv(file_path)
    width, height = 48, 48
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(width, height))
    images = np.stack(pixels, axis=0)
    images = np.expand_dims(images, -1)
    labels = to_categorical(data['emotion'], num_classes=7)
    return images, labels

# Create LSTM Sequences
def create_sequences(features, labels, seq_length=5):
    seqs, seq_labels = [], []
    for i in range(len(features) - seq_length):
        seqs.append(features[i : i + seq_length])
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(seqs), np.array(seq_labels)

# Build CNN

def build_cnn_functional():
    inputs = Input(shape=(48, 48, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs, name="cnn_model")
    return model

# Main
file_path = "fer2013.csv"
images, labels = load_data(file_path)
images = images / 255.0

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

cnn_model = build_cnn_functional()
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n[INFO] Training CNN...")
start_cnn = time.time()
cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=30,
    verbose=1
)
end_cnn = time.time()
print(f"\n CNN Training Time: {(end_cnn - start_cnn) / 60:.2f} minutes")
print_resource_usage("[After CNN Training]")

test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
print(f"\nCNN Test Accuracy: {test_acc * 100:.2f}%")
cnn_model.save("emotion_cnn.h5")
print("[SAVED] CNN -> 'emotion_cnn.h5'")

feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
print("\n[INFO] Extracting CNN features for LSTM...")
X_train_features = feature_extractor.predict(X_train, batch_size=64)
X_val_features   = feature_extractor.predict(X_val, batch_size=64)
X_test_features  = feature_extractor.predict(X_test, batch_size=64)

seq_length = 5
X_train_seq, y_train_seq = create_sequences(X_train_features, y_train, seq_length)
X_val_seq,   y_val_seq   = create_sequences(X_val_features,   y_val,   seq_length)
X_test_seq,  y_test_seq  = create_sequences(X_test_features,  y_test,  seq_length)

lstm_model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.3), input_shape=(seq_length, 256)),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, dropout=0.5)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n[INFO] Training LSTM...")
start_lstm = time.time()
lstm_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=32,
    verbose=1
)
end_lstm = time.time()
print(f"\n LSTM Training Time: {(end_lstm - start_lstm) / 60:.2f} minutes")
print_resource_usage("[After LSTM Training]")

lstm_loss, lstm_acc = lstm_model.evaluate(X_test_seq, y_test_seq)
print(f"\nLSTM Test Accuracy: {lstm_acc * 100:.2f}%")

np.save("X_test_lstm.npy", X_test_seq)
np.save("y_test_lstm.npy", y_test_seq)
lstm_model.save("emotion_lstm.h5")
print("[SAVED] LSTM -> 'emotion_lstm.h5'")
