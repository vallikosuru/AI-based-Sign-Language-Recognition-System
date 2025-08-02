import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
model_dict = pickle.load(open('C:/suprajaHackathon/sign-language-detector-python-master/sign-language-detector-python-master/model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Sign language label dictionary
labels_dict = {i: chr(65 + i) for i in range(26)}  # Auto-generate A-Z labels

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if webcam feed is unavailable

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract x, y coordinates
            for i in range(21):  # Only 21 landmarks needed
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize coordinates
            min_x, min_y = min(x_), min(y_)
            for i in range(21):  # Reduce feature count to 42
                data_aux.append(x_[i] - min_x)
                data_aux.append(y_[i] - min_y)

        # Ensure feature count matches model expectation
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "?")

            # Draw bounding box
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display output
    cv2.imshow('Sign Language Detector', frame)
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
