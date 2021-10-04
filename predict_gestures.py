import os
import random
import time

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from setup.app_properties import AppProperties as props

import pyautogui
# scientific ic to float notation
np.set_printoptions(suppress=True)


def get_gesture_data_from_hand_land_marks(hand_landmarks):
    row = []
    for item in hand_landmarks.landmark:
        row.append(item.x)
        row.append(item.y)
    return np.array(row).reshape(1, 42)


def predict_gesture_from_video(model, class_value_dict, gesture_gap_sec=0.5):
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    color = (255, 0, 0)
    thickness = 2
    last_gesture_time = int(time.time())
    last_gesture = "-1"
    message = ""

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
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
            # import pdb; pdb.set_trace()
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # import pdb; pdb.set_trace()
                    gesture_points = get_gesture_data_from_hand_land_marks(hand_landmarks)
                    predicted_gestures = model.predict(gesture_points)
                    message = pred_gesture = class_value_dict[np.where(predicted_gestures == predicted_gestures.max())[1][0]]
                    print(message)

                    # actions
                    print("Diff : {}".format(int(time.time()) - last_gesture_time))
                    curr_time = time.time()
                    if curr_time - last_gesture_time >= gesture_gap_sec:
                        if pred_gesture == "Thumbs Right":
                            print("==================>>> right")
                            pyautogui.hotkey('ctrl', 'tab')
                        elif pred_gesture == "Thumbs Left":
                            print("==================>>>left")
                            pyautogui.hotkey('ctrl', 'shift', 'tab')
                        # elif pred_gesture == "Fist":
                        #     print("fist")
                        #     pyautogui.hotkey('alt','tab')
                        elif pred_gesture == "Thumbs Up":
                            print("==================>>>pageup")
                            pyautogui.press('pageup')
                        elif pred_gesture == "Thumbs Down":
                            print("==================>>>pagedown")
                            pyautogui.press('pagedown')
                        last_gesture_time = time.time()

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            else:
                message = ""
            print(message)
            image = cv2.putText(image, message, org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


model_path = os.path.join(props.app_home, "models")
model = load_model(os.path.join(model_path))
rand_arr = np.array([random.uniform(0, 1) for i in range(42)]).reshape(1, 42)
pred = model.predict(rand_arr)
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
action_dict = {
 0
}
max_pred = pred.max()
pred_value = np.where(pred == np.amax(pred))[1]
print()


predict_gesture_from_video(model=model, class_value_dict=class_value_dict_rev)
