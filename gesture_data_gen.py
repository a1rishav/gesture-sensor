import mediapipe as mp
import cv2
import os
import time
import csv
from setup.app_properties import AppProperties as props

def get_gesture_data_from_hand_land_marks(hand_landmarks, gesture):
    row = []
    for item in hand_landmarks.landmark:
        row.append(item.x)
        row.append(item.y)
    row.append(gesture)
    return row

def write_data_to_csv(output_file_path, rows):
    with open(output_file_path, mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        for row in rows:
            (hand_landmarks, gesture) = row
            data_row = get_gesture_data_from_hand_land_marks(hand_landmarks, gesture)
            writer.writerow(data_row)

def extract_data_from_video(per_gesture_record_sec, gestures, output_file_path, inter_gesture_gap=3):
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (255, 0, 0)
    thickness = 2
    last_gesture_time = int(time.time())
    gesture_counter = -1
    message = ""
    inter_gesture_break = None
    gesture = None
    rows = []

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

        curr_time = int(time.time())
        if inter_gesture_break:
            gesture = None
            color = (255, 0, 0)
            if gesture_counter + 1 == len(gestures):
                break
            if curr_time - last_gesture_time <= inter_gesture_gap:
                message = "Next Gesture {} in : {}".format(gestures[gesture_counter + 1],
                                                           inter_gesture_gap - (curr_time - last_gesture_time))
            else:
                inter_gesture_break = False
                gesture_counter += 1
                last_gesture_time = int(time.time())
        elif curr_time - last_gesture_time <= per_gesture_record_sec:
            color = (0, 0, 255)
            if gesture_counter >= 0:
                gesture = gestures[gesture_counter]
                message = "Show gesture : {}, for : {}".format(gestures[gesture_counter],
                                                                 per_gesture_record_sec -
                                                                 (curr_time - last_gesture_time))
            else:
                message = "Starting recording gesture in : {}".format(per_gesture_record_sec -
                                                                      (curr_time - last_gesture_time))
        else:
            gesture = None
            inter_gesture_break = True
            last_gesture_time = int(time.time())

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
            if gesture:
                rows.append((hand_landmarks, gesture))
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        print(message)
        image = cv2.putText(image, message, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    write_data_to_csv(output_file_path, rows)

# data_path = os.path.join(props.data_dir, "train_gesture.csv")
# data_path = os.path.join(props.data_dir, "test_gesture.csv")
# extract_data_from_video(gestures=["Thumbs Up", "Thumbs Down", "Thumbs Right", "Thumbs Left", "Palm",
#                                   "Fist", "One Index", "Two Index"],
#                         per_gesture_record_sec=5, output_file_path=data_path)

data_path = os.path.join(props.data_dir, "sample_gesture.csv")
extract_data_from_video(gestures=["Thumbs Up", "Thumbs Down", "One Index", "Two Index", "Okay"],
                        per_gesture_record_sec=5, output_file_path=data_path)
