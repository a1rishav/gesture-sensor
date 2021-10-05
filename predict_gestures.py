import os
import time
import subprocess
import json

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pyautogui

from setup.app_properties import AppProperties as props

# configs
# scientific ic to float notation (used in models prediction)
np.set_printoptions(suppress=True)
pyautogui.FAILSAFE = False
class_value_dict_rev = {
    0: "Thumbs Up",
    1: "Thumbs Down",
    2: "Thumbs Right",
    3: "Thumbs Left",
    4: "Palm",
    5: "Fist",
    6: "One Index",
    7: "Two Index"
}

def get_gesture_data_from_hand_land_marks(hand_landmarks):
    row = []
    for item in hand_landmarks.landmark:
        row.append(item.x)
        row.append(item.y)
    return np.array(row).reshape(1, 42)

def get_active_app(apps):
    result = subprocess.run(['xdotool', 'getwindowfocus', 'getwindowname'], stdout=subprocess.PIPE)
    out = str(result.stdout)
    print(out)
    for app in apps:
        if app in out.lower():
            return app
    return None

def get_app_action_dict(apps, app_gesture_action_dict):
    app = get_active_app(apps)
    if app:
       return app_gesture_action_dict[app]

def predict_gesture_from_video(model, class_value_dict, app_gesture_action_dict,
                               gesture_gap_sec=0.5, gesture_confidence_threshold=0.9):
    # opencv configs
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (255, 0, 0)
    thickness = 3
    last_gesture_time = int(time.time())
    message = ""
    app = None
    pred_gesture = None

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)


    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    curr_time = time.time()
                    if curr_time - last_gesture_time < gesture_gap_sec:
                        print("Inter gesture gap hasn't crossed: {}")
                        continue

                    gesture_points = get_gesture_data_from_hand_land_marks(hand_landmarks)
                    predicted_gestures = model.predict(gesture_points)
                    last_gesture_time = time.time()
                    predicted_gesture_confidence = predicted_gestures.max()

                    if predicted_gesture_confidence < gesture_confidence_threshold:
                        print("Skipping..., gesture confidence : {}".format(predicted_gesture_confidence))
                        continue

                    message = pred_gesture = class_value_dict[
                        np.where(predicted_gestures == predicted_gesture_confidence)[1][0]]
                    print(message)

                    if pred_gesture == "Fist":
                        pyautogui.keyDown('alt')
                        pyautogui.press('tab')
                        continue

                    action_dict = app_gesture_action_dict['common'].copy()
                    if app:
                        action_dict.update(app_gesture_action_dict[app])

                    print("App : {}, Action Dict : {}".format(app, action_dict))
                    if pred_gesture not in action_dict:
                        continue

                    action = action_dict[pred_gesture]
                    print("Action : {}".format(action))

                    if isinstance(action, str):
                        pyautogui.press(action)
                    elif isinstance(action, list):
                        pyautogui.hotkey(*tuple(action))
            else:
                if pred_gesture == "Fist":
                    pyautogui.keyUp('alt')
                    app = get_active_app(apps=app_gesture_action_dict.keys())
                    pred_gesture = None
                message = ""

            if message:
                print(message)
            image = cv2.putText(image, message, org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            image = cv2.resize(image, (300, 300))
            cv2.imshow('Hand Gesture Recognition', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

# load model
model_path = os.path.join(props.app_home, "models")
model = load_model(os.path.join(model_path))

# load app-gesture dict
with open(props.gesture_config) as json_file:
    gesture_config = json.load(json_file)

predict_gesture_from_video(model=model, class_value_dict=class_value_dict_rev,
                           app_gesture_action_dict=gesture_config)
