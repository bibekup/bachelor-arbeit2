#realtime_enotion_using_cnn.py
import os
# Suppress excessive TensorFlow logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
from tensorflow.keras.models import load_model
#from demo import model
from tensorflow.keras.preprocessing.image import img_to_array
# Load model
model = load_model('emotion_detection_model.h5', compile=False)
# Match your model's output order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Could not open webcam.')
    exit()
# Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
max_frames = 5       # We want exactly 5 frames where faces are detected
face_frames_count = 0  # Counts only frames with faces
# Time (in ms) to display each frame
delay_ms = 1000
while face_frames_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )
    # Only process further if we have faces
    if len(faces) > 0:
        # Increase count for a valid face frame
        face_frames_count += 1
        print(f"Frame with face #{face_frames_count} of {max_frames}")
        for (x, y, w, h) in faces:
            # Preprocess ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)
            # Predict
            prediction = model.predict(face_roi, verbose=0)
            label_idx = np.argmax(prediction)
            emotion = emotion_labels[label_idx]
            # Draw bounding box & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )
            print(f"Detected emotion -> {emotion}")
    # Display
    cv2.imshow("Emotion Detection (Faces Only Count)", frame)

    # Wait for the delay or 'q' to exit
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord('q'):
        print("User pressed 'q'; exiting early.")
        break
# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Done! Collected 5 frames with faces (or exited early).")