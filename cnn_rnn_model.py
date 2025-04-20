#cnn_rnn_model.py
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, SimpleRNN)
from tensorflow.keras.utils import to_categorical

# Load and preprocess FER-2013 data
def load_data(file_path):
    data = pd.read_csv(file_path)
    width, height = 48, 48
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(width, height))
    images = np.stack(pixels, axis=0)
    images = np.expand_dims(images, -1)
    labels = to_categorical(data['emotion'], num_classes=7)
    return images / 255.0, labels

# Create sequences for RNN
def create_sequences(features, labels, seq_length=5):
    seqs, seq_labels = [], []
    for i in range(len(features) - seq_length):
        seqs.append(features[i : i + seq_length])
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(seqs), np.array(seq_labels)

# CNN feature extractor (same as before)
def build_cnn_feature_extractor():
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
    model = Model(inputs=inputs, outputs=x)
    return model

# Main execution
file_path = "fer2013.csv"
images, labels = load_data(file_path)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Feature extraction with CNN
cnn_model = build_cnn_feature_extractor()
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_feat = cnn_model.predict(X_train, batch_size=64)
X_val_feat = cnn_model.predict(X_val, batch_size=64)
X_test_feat = cnn_model.predict(X_test, batch_size=64)

# Create sequences for RNN
seq_length = 5
X_train_seq, y_train_seq = create_sequences(X_train_feat, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val_feat, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test_feat, y_test, seq_length)

# Build RNN model
rnn_model = Sequential([
    SimpleRNN(128, input_shape=(seq_length, 256), return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train RNN
print("\n[INFO] Training RNN...")
start_time = time.time()
rnn_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=30,
    batch_size=32,
    verbose=1
)
end_time = time.time()
print(f"\n🕒 RNN Training Time: {(end_time - start_time) / 60:.2f} minutes")

# Evaluation
loss, accuracy = rnn_model.evaluate(X_test_seq, y_test_seq)
print(f"\n✅ RNN Test Accuracy: {accuracy * 100:.2f}%")

