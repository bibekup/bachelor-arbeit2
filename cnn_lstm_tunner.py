#cnn_lstm_tunner.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, LSTM, Bidirectional)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------------------------------------------
# 1) Load FER-2013
# -----------------------------------------------------------------
def load_data(file_path):
    data = pd.read_csv(file_path)
    w, h = 48, 48

    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype=np.uint8).reshape(w, h))
    images = np.stack(pixels, axis=0)[..., np.newaxis] / 255.0  # shape => (N,48,48,1)
    labels = to_categorical(data['emotion'], num_classes=7)     # shape => (N,7)
    return images, labels


# -----------------------------------------------------------------
# 2) Build CNN (Functional)
# -----------------------------------------------------------------
def build_cnn():
    inputs = Input(shape=(48, 48, 1))
    x = Conv2D(32, 3, activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 3, activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------------------------------------------
# 3) Create sequences for LSTM
# -----------------------------------------------------------------
def create_sequences(features, labels, seq_len=5):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i : i + seq_len])
        y_seq.append(labels[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)


# -----------------------------------------------------------------
# 4) Build LSTM model (for scikeras)
# -----------------------------------------------------------------
def build_lstm_model(lstm_units=128, dropout_rate=0.5, input_shape=(5, 256)):
    model = Sequential([
        Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate), input_shape=input_shape),
        Bidirectional(LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate)),
        Bidirectional(LSTM(lstm_units // 4, dropout=dropout_rate)),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ---------- MAIN SCRIPT ----------
if __name__ == "__main__":

    # Load data & split
    X, y = load_data("fer2013.csv")  # Adjust path if needed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train CNN
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    cnn_model = build_cnn()
    cnn_model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        epochs=5,
        verbose=1
    )

    # Extract features from second-to-last layer
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    train_feats = feature_extractor.predict(X_train, batch_size=64)
    test_feats  = feature_extractor.predict(X_test, batch_size=64)

    # Create LSTM sequences
    seq_len = 5
    X_train_seq, y_train_seq = create_sequences(train_feats, y_train, seq_len)
    X_test_seq,  y_test_seq  = create_sequences(test_feats,  y_test,  seq_len)

    # Grid Search for LSTM hyperparams
    lstm_wrapper = KerasClassifier(
        model=build_lstm_model,
        model__input_shape=(seq_len, train_feats.shape[1]),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    param_grid = {
        "model__lstm_units": [64, 128],
        "model__dropout_rate": [0.3, 0.5],
        "epochs": [5, 10]
    }

    grid_search = GridSearchCV(
        estimator=lstm_wrapper,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        verbose=2,
        error_score="raise"
    )

    print("\n[INFO] Grid search for LSTM...")
    grid_search.fit(X_train_seq, y_train_seq)

    print("\n[RESULT] Best Params:", grid_search.best_params_)
    print(f"[RESULT] Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

    #  Show all fold results
    results_df = pd.DataFrame(grid_search.cv_results_)
    cols_to_display = [
        'params',
        'mean_test_score',
        'std_test_score',
        'rank_test_score',
        'split0_test_score',
        'split1_test_score',
        'split2_test_score'
    ]
    print("\n[DETAILS] Grid Search All Fold Results:\n")
    print(results_df[cols_to_display].sort_values("rank_test_score"))

    # Save results to CSV
    results_df.to_csv("lstm_gridsearch_results.csv", index=False)
    print("\n[SAVED] Grid search results saved to 'lstm_gridsearch_results.csv'")

    # G) Final Evaluation
    best_lstm = grid_search.best_estimator_
    y_pred = best_lstm.predict(X_test_seq)

    test_acc = accuracy_score(np.argmax(y_test_seq, axis=1), y_pred)
    print(f"[INFO] Test Accuracy of Best LSTM: {test_acc*100:.2f}%")

    #  Save the best model
    # best_lstm.model_.save("cnn_lstm_grid_tuned.h5")
