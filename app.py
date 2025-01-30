import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/sign_language_mode.keras'
model = load_model(MODEL_PATH)

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Label classes (A-Z)
labels = [chr(i) for i in range(65, 91)]

# Helper function to preprocess frame
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Pad or trim to fixed size
            max_landmarks = 21
            data_aux = data_aux[:max_landmarks * 2] + [0] * (max_landmarks * 2 - len(data_aux))
            data_aux = np.array(data_aux).reshape(1, len(data_aux), 1, 1, 1).astype('float32')
            data_aux /= np.max(data_aux)
            return data_aux

    return None

# Video streaming generator
def video_stream():
    cap = cv2.VideoCapture(0)  # Access webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
        processed_data = preprocess_frame(frame)

        if processed_data is not None:
            prediction = model.predict(processed_data)
            predicted_class = labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Display prediction on the frame
            cv2.putText(frame, f"{predicted_class} ({confidence*100:.2f}%)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
