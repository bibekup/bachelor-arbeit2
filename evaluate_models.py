# evaluate_models.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load CNN Test Data
X_test_cnn = np.load("X_test_cnn.npy")
y_test_cnn = np.load("y_test_cnn.npy")

# Load LSTM Test Sequences
X_test_lstm_seq = np.load("X_test_lstm.npy")
y_test_lstm_seq = np.load("y_test_lstm.npy")

# Load trained models
cnn_model = load_model("emotion_detection_model.h5")
lstm_model = load_model("emotion_lstm.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 1. CNN Evaluation
print("\nEvaluating CNN Model:\n")
cnn_preds = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
y_test_cnn_labels = np.argmax(y_test_cnn, axis=1)

cnn_accuracy = np.mean(cnn_preds == y_test_cnn_labels)
print(f"CNN Accuracy: {cnn_accuracy * 100:.2f}%")

print("CNN Classification Report:")
print(classification_report(y_test_cnn_labels, cnn_preds, target_names=emotion_labels))

cnn_cm = confusion_matrix(y_test_cnn_labels, cnn_preds)
plt.figure(figsize=(10,7))
sns.heatmap(cnn_cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# LSTM Evaluation
print("\nEvaluating LSTM Model:\n")
lstm_preds = np.argmax(lstm_model.predict(X_test_lstm_seq), axis=1)
y_test_lstm_labels = np.argmax(y_test_lstm_seq, axis=1)

lstm_accuracy = np.mean(lstm_preds == y_test_lstm_labels)
print(f"LSTM Accuracy: {lstm_accuracy*100:.2f}%")

# Classification report
print("\nLSTM Classification Report:")
print(classification_report(y_test_lstm_labels, lstm_preds, target_names=emotion_labels))

# Confusion matrix
lstm_cm = confusion_matrix(y_test_lstm_labels, lstm_preds)
plt.figure(figsize=(10,7))
sns.heatmap(lstm_cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Blues')
plt.title('LSTM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Inference Timing
print("\nInference Timing:")
start = tf.timestamp()
cnn_model.predict(X_test_cnn)
cnn_inference_time = tf.timestamp() - start
print(f"CNN Inference Time: {cnn_inference_time:.4f} seconds")

start = tf.timestamp()
lstm_model.predict(X_test_lstm_seq)
lstm_inference_time = tf.timestamp() - start
print(f"LSTM Inference Time: {lstm_inference_time:.4f} seconds")
