#graphical_evaluation.py
import matplotlib.pyplot as plt
import numpy as np

# Emotions labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# CNN metrics
cnn_precision = [0.48, 0.53, 0.44, 0.76, 0.49, 0.68, 0.54]
cnn_recall = [0.51, 0.10, 0.33, 0.86, 0.45, 0.72, 0.58]
cnn_f1 = [0.50, 0.17, 0.38, 0.81, 0.47, 0.70, 0.56]

# CNN-LSTM metrics
lstm_precision = [0.50, 0.40, 0.44, 0.75, 0.42, 0.65, 0.53]
lstm_recall = [0.44, 0.36, 0.20, 0.83, 0.58, 0.73, 0.52]
lstm_f1 = [0.47, 0.38, 0.27, 0.79, 0.49, 0.69, 0.53]

x = np.arange(len(emotions))
width = 0.25

# Plot for CNN Model
plt.figure(figsize=(14, 6))
plt.bar(x - width, cnn_precision, width, label='Precision')
plt.bar(x, cnn_recall, width, label='Recall')
plt.bar(x + width, cnn_f1, width, label='F1-Score')
plt.xticks(x, emotions)
plt.title("CNN Model: Precision, Recall, and F1-Score per Emotion")
plt.xlabel("Emotions")
plt.ylabel("Scores")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y')
plt.show()

# Plot for CNN-LSTM Model
plt.figure(figsize=(14, 6))
plt.bar(x - width, lstm_precision, width, label='Precision')
plt.bar(x, lstm_recall, width, label='Recall')
plt.bar(x + width, lstm_f1, width, label='F1-Score')
plt.xticks(x, emotions)
plt.title("CNN-LSTM Model: Precision, Recall, and F1-Score per Emotion")
plt.xlabel("Emotions")
plt.ylabel("Scores")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y')
plt.show()
