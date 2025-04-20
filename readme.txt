# Emotion Detection using CNN, CNN-LSTM

This project implements facial emotion detection using deep learning models (CNN, CNN-LSTM, CNN-RNN) trained on the FER-2013 dataset. It supports training, evaluation, hyperparameter tuning, and real-time inference from a webcam.

## 📁 Project Structure

    - `fer2013.csv`:                        Dataset used for training and testing.

    - `cnn_model.py`:                       Builds and trains a CNN model for emotion classification.

    - `cnn_lstm_model.py`:                  Trains a CNN followed by LSTM for temporal emotion recognition.

    - `cnn_rnn_model.py`:                   Combines CNN and RNN for feature extraction and sequence learning.

    - `cnn_tunner.py`:                      Hyperparameter tuning for CNN using GridSearchCV and k-fold cross-validation.

    - `cnn_lstm_tunner.py`:                 Hyperparameter tuning for CNN-LSTM architecture and and k-fold cross-validation.

    - `evaluate_models.py`:                 Loads saved models and evaluates on the basis of perfermence metrix.

    - `graphical_evaluation.py`:            Plots precision, recall, and F1-score per emotion.

    - `piechart.py`:                        Visualizes dataset and confusion matrix as pie chart.

    - `realtime_emotion_using_cnn.py`:      Real-time emotion detection using a webcam using  CNN Model.

    - `realtime_emotion_using_lstm.py`:     Real-time emotion detection using CNN feature extractor and LSTM Model.

    - `emotion_cnn.h5`, 
    `emotion_lstm.h5`, `emotion_rnn.h5`:    Trained model files.

    ## Emotions Supported

    - Angry
    - Disgust
    - Fear
    - Happy
    - Sad
    - Surprise
    - Neutral

##  How to Run

    1. **Install dependencies:**
    2. **Train models:**
        - CNN: `python cnn_model.py`
        - CNN-LSTM: `python cnn_lstm_model.py`

    3. **Real-time detection:**
        - CNN only: `python realtime_emotion_using_cnn.py`
        - CNN-LSTM: `python realtime_emotion_using_lstm.py`

    4. **Evaluation and tunning is optional**  


  
## Python Environments

    This project uses **two Python virtual environments** for better modularity:

    1. `tf-env` (TensorFlow environment)

    Used for TensorFlow-based deep learning scripts:
    - `cnn_model.py`
    - `cnn_lstm_model.py`
    - `cnn_rnn_model.py`
    - `evaluate_models.py`
    - `realtime_emotion_using_cnn.py`
    - `realtime_emotion_using_lstm.py`

    2. venv310
    used for auxiliary tasks like plotting or tuning.
    - ´graphical_evaluation.py`
    - `piechart.py`
    - `cnn_tunner.py`
    - `cnn_lstm_tunner.py`

