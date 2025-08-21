import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import pygame
from collections import deque  # For buffering positions

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)  # Keep 30 FPS for responsiveness

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.6,  # Increased for better detection reliability
    min_tracking_confidence=0.6,   # Increased for better tracking
    model_complexity=0  # Lightweight model for performance
)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables for right hand and leg
previous_wrist_y = None
previous_wrist_x = None
punch_threshold = 20  # Increased to reduce false positives

previous_knee_y = None
previous_knee_x = None
kick_threshold = 25  # Increased to reduce false positives

# Variables for left hand and leg
previous_left_wrist_y = None
previous_left_wrist_x = None
previous_left_knee_y = None
previous_left_knee_x = None

# Smoothed positions for noise reduction
smoothed_wrist_y = None
smoothed_wrist_x = None
smoothed_knee_y = None
smoothed_knee_x = None
smoothed_left_wrist_y = None
smoothed_left_wrist_x = None
smoothed_left_knee_y = None
smoothed_left_knee_x = None

# Smoothing factor and buffer
alpha = 0.5  # Adjusted for balanced responsiveness and smoothness
buffer_size = 5  # Increased buffer size for better noise filtering
wrist_y_buffer = deque(maxlen=buffer_size)
wrist_x_buffer = deque(maxlen=buffer_size)
knee_y_buffer = deque(maxlen=buffer_size)
knee_x_buffer = deque(maxlen=buffer_size)
left_wrist_y_buffer = deque(maxlen=buffer_size)
left_wrist_x_buffer = deque(maxlen=buffer_size)
left_knee_y_buffer = deque(maxlen=buffer_size)
left_knee_x_buffer = deque(maxlen=buffer_size)

action_in_progress = False
frame_counter = 0
last_results = None
last_action_time = 0
action_cooldown = 0.3  # Increased cooldown to 300ms to prevent rapid triggers

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

print("Starting... Move your right or left arm to punch or lift your right or left leg to kick!")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No player detected in front of IronVision")
            break

        frame_counter += 1
        current_time = time.time()

        # Process pose detection every frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            last_results = results

        # Use the last known landmarks for detection
        if last_results and last_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, last_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            wrist = last_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            wrist_y = wrist.y * frame.shape[0]
            wrist_x = wrist.x * frame.shape[1]

            knee = last_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            knee_y = knee.y * frame.shape[0]
            knee_x = knee.x * frame.shape[1]

            # Left side landmarks
            left_wrist = last_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            left_wrist_y = left_wrist.y * frame.shape[0]
            left_wrist_x = left_wrist.x * frame.shape[1]

            left_knee = last_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            left_knee_y = left_knee.y * frame.shape[0]
            left_knee_x = left_knee.x * frame.shape[1]

            # Smooth the positions
            if smoothed_wrist_y is None:
                smoothed_wrist_y = wrist_y
                smoothed_wrist_x = wrist_x
                smoothed_knee_y = knee_y
                smoothed_knee_x = knee_x
                smoothed_left_wrist_y = left_wrist_y
                smoothed_left_wrist_x = left_wrist_x
                smoothed_left_knee_y = left_knee_y
                smoothed_left_knee_x = left_knee_x
            else:
                smoothed_wrist_y = alpha * wrist_y + (1 - alpha) * smoothed_wrist_y
                smoothed_wrist_x = alpha * wrist_x + (1 - alpha) * smoothed_wrist_x
                smoothed_knee_y = alpha * knee_y + (1 - alpha) * smoothed_knee_y
                smoothed_knee_x = alpha * knee_x + (1 - alpha) * smoothed_knee_x
                smoothed_left_wrist_y = alpha * left_wrist_y + (1 - alpha) * smoothed_left_wrist_y
                smoothed_left_wrist_x = alpha * left_wrist_x + (1 - alpha) * smoothed_left_wrist_x
                smoothed_left_knee_y = alpha * left_knee_y + (1 - alpha) * smoothed_left_knee_y
                smoothed_left_knee_x = alpha * left_knee_x + (1 - alpha) * smoothed_left_knee_x

            # Add positions to buffers
            wrist_y_buffer.append(smoothed_wrist_y)
            wrist_x_buffer.append(smoothed_wrist_x)
            knee_y_buffer.append(smoothed_knee_y)
            knee_x_buffer.append(smoothed_knee_x)
            left_wrist_y_buffer.append(smoothed_left_wrist_y)
            left_wrist_x_buffer.append(smoothed_left_wrist_x)
            left_knee_y_buffer.append(smoothed_left_knee_y)
            left_knee_x_buffer.append(smoothed_left_knee_x)

            # Only process actions if not in cooldown
            if not action_in_progress and (current_time - last_action_time) > action_cooldown:
                # Check for right punch
                if len(wrist_y_buffer) == buffer_size and len(wrist_x_buffer) == buffer_size:
                    max_wrist_y_diff = max(wrist_y_buffer) - min(wrist_y_buffer)
                    max_wrist_x_diff = max(wrist_x_buffer) - min(wrist_x_buffer)
                    max_total_movement = (max_wrist_x_diff**2 + max_wrist_y_diff**2)**0.5

                    if max_total_movement > punch_threshold and max_wrist_y_diff > 10:
                        action_in_progress = True
                        last_action_time = current_time
                        print("Right Punch detected")
                        pyautogui.press("p")
                        cv2.putText(frame, "Right Punch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        action_in_progress = False

                # Check for right kick
                if len(knee_y_buffer) == buffer_size and len(knee_x_buffer) == buffer_size:
                    max_knee_y_diff = max(knee_y_buffer) - min(knee_y_buffer)
                    max_knee_x_diff = max(knee_x_buffer) - min(knee_x_buffer)
                    max_total_knee_movement = (max_knee_x_diff**2 + max_knee_y_diff**2)**0.5

                    if max_total_knee_movement > kick_threshold and max_knee_y_diff < -10:
                        action_in_progress = True
                        last_action_time = current_time
                        print("Right Kick detected")
                        pyautogui.press("k")
                        cv2.putText(frame, "Right Kick", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        action_in_progress = False

                # Check for left punch
                if len(left_wrist_y_buffer) == buffer_size and len(left_wrist_x_buffer) == buffer_size:
                    max_left_wrist_y_diff = max(left_wrist_y_buffer) - min(left_wrist_y_buffer)
                    max_left_wrist_x_diff = max(left_wrist_x_buffer) - min(left_wrist_x_buffer)
                    max_left_total_movement = (max_left_wrist_x_diff**2 + max_left_wrist_y_diff**2)**0.5

                    if max_left_total_movement > punch_threshold and max_left_wrist_y_diff > 10:
                        action_in_progress = True
                        last_action_time = current_time
                        print("Left Punch detected")
                        pyautogui.press("p")
                        cv2.putText(frame, "Left Punch", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        action_in_progress = False

                # Check for left kick
                if len(left_knee_y_buffer) == buffer_size and len(left_knee_x_buffer) == buffer_size:
                    max_left_knee_y_diff = max(left_knee_y_buffer) - min(left_knee_y_buffer)
                    max_left_knee_x_diff = max(left_knee_x_buffer) - min(left_knee_x_buffer)
                    max_left_total_knee_movement = (max_left_knee_x_diff**2 + max_left_knee_y_diff**2)**0.5

                    if max_left_total_knee_movement > kick_threshold and max_left_knee_y_diff < -10:
                        action_in_progress = True
                        last_action_time = current_time
                        print("Left Kick detected")
                        pyautogui.press("k")
                        cv2.putText(frame, "Left Kick", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        action_in_progress = False

            # Update previous positions
            previous_wrist_y = smoothed_wrist_y
            previous_wrist_x = smoothed_wrist_x
            previous_knee_y = smoothed_knee_y
            previous_knee_x = smoothed_knee_x
            previous_left_wrist_y = smoothed_left_wrist_y
            previous_left_wrist_x = smoothed_left_wrist_x
            previous_left_knee_y = smoothed_left_knee_y
            previous_left_knee_x = smoothed_left_knee_x

        # Display the video feed every 2 frames
        if frame_counter % 2 == 0:
            cv2.imshow("WebCam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up and exit
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Script exited successfully.")