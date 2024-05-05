import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'index up', 1: 'index down', 2: 'thumbs up', 3: 'Four fingers', 4: 'middle up', 5: 'middle down'
}

# Action mapping for Instagram
action_mapping = {
    'index up': 'ScrollUp',
    'index down': 'ScrollDown',
    'thumbs up': 'Like/Unlike',
    'Four fingers': 'Story',
    'middle up': 'InstagramIcon',
    'middle down': 'Reels',
}

# Define the allowed gestures
allowed_gestures = list(action_mapping.keys())

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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
            # Predict
            prediction = model.predict([np.asarray(data_aux)])
            predicted_gesture = labels_dict[int(prediction[0])]
            
            # Check if predicted gesture is allowed
            if predicted_gesture in allowed_gestures:
                action = action_mapping.get(predicted_gesture)

                # Perform action based on gesture
                if action == 'ScrollUp':
                    pyautogui.scroll(50)  # Scroll up
                elif action == 'ScrollDown':
                    pyautogui.scroll(-40)  # Scroll down
                elif action == 'Like/Unlike':
                    # Click on the like/Unlike button (replace coordinates)
                    pyautogui.click(x=395, y=900)  # Adjust coordinates as needed
                elif action == 'InstagramIcon':
                    # Click on the InstagramIcon button (replace coordinates)
                    pyautogui.click(x=50, y=80)  # Adjust coordinates as needed
                elif action == 'Story':
                    # Click on the Story button (replace coordinates)
                    pyautogui.click(x=350, y=90)  # Adjust coordinates as needed
                elif action == 'Reels':
                    # Click on the Reels button (replace coordinates)
                    pyautogui.click(x=70, y=400)  # Adjust coordinates as needed

        except Exception as e:
            print('Error:', e)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_gesture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
