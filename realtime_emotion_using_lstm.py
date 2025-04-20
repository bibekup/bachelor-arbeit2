#realtime_emotion_using_lstm.py
import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

def real_time_detection(
    cnn_path="emotion_cnn.h5",
    lstm_path="emotion_lstm.h5",
    use_face_detection=True
):

    print("[INFO] Loading models...")
    cnn_model = load_model(cnn_path)
    lstm_model = load_model(lstm_path)

    feature_extractor = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[-2].output
    )

    if use_face_detection:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    else:
        face_cascade = None

    SEQ_LENGTH = 5
    feature_buffer = deque(maxlen=SEQ_LENGTH)
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("Starting real-time detection. Press 'q' to quit.")

    last_emotion = None
    display_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_cascade:
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30)
            )
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
            else:
                cv2.imshow("Real-Time Emotion", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            resized = cv2.resize(face_roi, (48, 48))
        else:
            resized = cv2.resize(gray, (48,48))

        input_img = resized.astype("float32") / 255.0
        input_img = np.expand_dims(input_img, axis=-1)
        input_img = np.expand_dims(input_img, axis=0)

        feats = feature_extractor.predict(input_img, verbose=0)[0]
        feature_buffer.append(feats)

        if len(feature_buffer) == SEQ_LENGTH:
            lstm_input = np.array(feature_buffer).reshape(1, SEQ_LENGTH, 256)
            preds = lstm_model.predict(lstm_input, verbose=0)
            emotion_idx = np.argmax(preds[0])
            current_emotion = emotion_labels[emotion_idx]

            # Handle transitions
            if last_emotion is None:
                display_text = current_emotion
            elif current_emotion != last_emotion:
                display_text = f"{last_emotion} -> {current_emotion}"
            else:
                display_text = current_emotion

            last_emotion = current_emotion

            # Draw text on screen
            cv2.putText(
                frame,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        cv2.imshow("Real-Time Emotion", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting real-time detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# USAGE
# ---------------------------
if __name__ == "__main__":
    real_time_detection(
        cnn_path="emotion_cnn.h5",
        lstm_path="emotion_lstm.h5",
        use_face_detection=True
    )