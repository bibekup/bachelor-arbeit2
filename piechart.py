#piechart.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the FER2013 dataset
data = pd.read_csv('fer2013.csv')

# Emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Create a figure to display one image per emotion
plt.figure(figsize=(12, 6))
for emotion, label in emotion_labels.items():
    # Select the first image for each emotion
    img_data = data[data['emotion'] == emotion].iloc[0]['pixels']
    img = np.fromstring(img_data, sep=' ').reshape(48, 48)
    plt.subplot(2, 4, emotion + 1)
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()


# Confusion matrix data (rows = true labels)
confusion_matrix = [
    [321, 24, 41, 57, 166, 32, 92],   # Angry
    [23, 29, 4, 3, 12, 4, 5],         # Disgust
    [104, 9, 149, 67, 230, 125, 77],  # Fear
    [39, 3, 19, 1108, 56, 38, 71],    # Happy
    [79, 6, 44, 74, 522, 14, 167],    # Sad
    [17, 0, 44, 53, 21, 440, 28],     # Surprise
    [63, 1, 36, 106, 235, 19, 502]    # Neutral
]

# Emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Calculate total samples per true emotion class
total_data_distribution = [sum(row) for row in confusion_matrix]

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(total_data_distribution, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Total Data Distribution by Emotion Labels")
plt.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
plt.show()
