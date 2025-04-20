#cnn_tunner.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scikeras.wrappers import KerasClassifier

# Load and preprocess the FER-2013 dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    width, height = 48, 48
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(width, height))
    images = np.stack(pixels, axis=0)
    images = np.expand_dims(images, -1)
    labels = to_categorical(data['emotion'])
    return images / 255.0, labels

# Define a function to build the CNN model
def build_model(conv_filters_1=32, conv_filters_2=64, dense_units=256, dropout_rate=0.5):
    model = Sequential([
        Conv2D(conv_filters_1, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(conv_filters_2, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main block to load data and tune model
if __name__ == "__main__":
    # Load data
    file_path = 'fer2013.csv'
    X, y = load_data(file_path)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model for scikit-learn
    model = KerasClassifier(
        model=build_model,
        epochs=5,  # Keep small for tuning
        batch_size=64,
        verbose=1
    )

    # Define hyperparameter grid
    param_grid = {
        "model__conv_filters_1": [32, 64],
        "model__conv_filters_2": [64, 128],
        "model__dense_units": [128, 256],
        "model__dropout_rate": [0.3, 0.5]
    }

    # Grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
    grid.fit(X_train, y_train)

    # Output best parameters
    print("\n✅ Best Parameters:", grid.best_params_)
    print("✅ Best Accuracy:", f"{grid.best_score_ * 100:.2f}%")
