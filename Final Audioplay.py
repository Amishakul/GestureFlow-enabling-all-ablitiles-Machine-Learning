import cv2
import mediapipe as mp
import pyautogui
import pickle
import time
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the model
model_path = r'C:\amisha ml\model.p'

model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
mp_hands_constants = mp.solutions.hands.HandLandmark

labels_dict = {
    11: 'palm', 12: 'index in right', 13: 'index in left', 14: 'Three fingers', 15: 'Three Finger in right'
}

# Action mapping for gesture control
action_mapping = {
    'palm': 'Play',
    'index in right': 'Next_song',
    'index in left': 'Previous_song',
    'Three fingers': 'Increase_Volume',
    'Three Finger in right': 'Reduce_volume',
}

last_action_time = time.time()
cooldown_duration = 5      # Adjust the cooldown duration as needed (in seconds)

allowed_gestures = list(action_mapping.keys())

def gesture_command():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                predicted_gesture = 'None'  # Default value

                try:
                    # Predict gesture
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_gesture = labels_dict[int(prediction[0])]

                    if predicted_gesture in allowed_gestures:
                        action = action_mapping[predicted_gesture]

                        if action == 'Play':
                            # Perform Play action (Example: spacebar for play/pause)
                            pyautogui.press('space')

                        elif action == 'Next_song':
                            # Perform Next song action (Example: Ctrl + F)
                            pyautogui.hotkey('ctrl', 'f')

                        elif action == 'Previous_song':
                            # Perform Previous song action (Example: Ctrl + B)
                            pyautogui.hotkey('ctrl', 'b')

                        elif action == 'Increase_Volume':
                            # Perform Increase volume action (Example: F3)
                            pyautogui.press('volumeup')
                            time.sleep(1)  # Optional: Adjust if needed

                        elif action == 'Reduce_volume':
                            # Perform Reduce volume action (Example: F2)
                            pyautogui.press('volumedown')
                            time.sleep(1)  # Optional: Adjust if needed

                        last_action_time = time.time()

                except Exception as e:
                    print('Error:', e)

        cv2.imshow('Gesture Detection for Media Player Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_command()
