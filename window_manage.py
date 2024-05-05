import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import keyboard
import os
import time  # Import the time module for delays

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
    6: 'palm',  8: 'Thumb right', 17: 'pinky horiz', 10: 'say Cheese', 18: 'Yo sign'
}

# Action mapping for Instagram
action_mapping = {
    'palm': 'MinimizeWindow', 
    'pinky horiz': 'Switch tab', 
    'say Cheese': 'FileExplorer', 
    'Yo sign': 'NewFolder', 
    'Thumb right': 'Window'
}

# Define the allowed gestures
allowed_gestures = list(action_mapping.keys())

def create_folder(drive_path, folder_name):
    # Combine the drive path and folder name to create the full path
    folder_path = os.path.join(drive_path, folder_name)

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        try:
            # Create the new folder
            os.makedirs(folder_path)
            print(f"Folder '{folder_name}' created successfully in '{drive_path}'")
        except OSError as e:
            print(f"Error creating folder: {e}")
    else:
        print(f"Folder '{folder_name}' already exists in '{drive_path}'")

last_switch_time = time.time()  # Initialize the last switch time
switch_interval = 3  # Define the interval in seconds for switching tabs

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

                if action == 'MinimizeWindow':
                    pyautogui.hotkey('win', 'down')  # Minimize the current window
                elif action == 'Switch tab':
                    current_time = time.time()
                    if current_time - last_switch_time >= switch_interval:
                        pyautogui.hotkey('alt', 'tab')   # Switch tab  
                        last_switch_time = current_time  # Update the last switch time                
                elif action == 'FileExplorer':
                    pyautogui.hotkey('win', 'e')  # File Explorer

                elif action == 'NewFolder':
                    # Press Ctrl + Shift + N keyboard shortcut to create new folder
                    keyboard.send('ctrl+shift+n')
                    print("Ctrl + Shift + N pressed to create new folder")

                    # Replace 'C:\\' with the desired drive path and 'NewFolder' with the desired folder name
                    drive_path = 'C:\\'  
                    folder_name = 'NewFolder'
                    create_folder(drive_path, folder_name)

                elif action == 'Window':
                    pyautogui.press('win')
                

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